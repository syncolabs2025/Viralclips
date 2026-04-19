"""
Per-shot first-frame analysis.

Detection stack (each layer falls back gracefully):
  1. InsightFace buffalo_l — face detection + 512-d ArcFace embeddings
  2. GroundingDINO         — open-vocabulary object detection for non-face subjects
  3. Qwen2-VL              — local VLM; fires only for 3+ faces or ambiguous shots

Output: ShotAnalysisResult — shot_type, detected faces/objects, primary subject
bbox, and a suggested LayoutType string that the engine may override.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("reframe.shot_analysis")


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class FaceResult:
    bbox:        np.ndarray   # [x1, y1, x2, y2] pixel coords in source frame
    embedding:   np.ndarray   # 512-d ArcFace embedding (unit-normalised)
    cx:          float
    cy:          float
    area:        float        # pixel area of bbox
    center_dist: float        # normalised distance from frame centre [0, 1]


@dataclass
class ObjectResult:
    bbox:  np.ndarray         # [x1, y1, x2, y2] pixel coords
    label: str
    score: float
    cx:    float
    cy:    float


@dataclass
class ShotAnalysisResult:
    shot_type:            str                           # solo|dual|group|screen|scenic|pip
    faces:                list[FaceResult]  = field(default_factory=list)
    objects:              list[ObjectResult] = field(default_factory=list)
    primary_subject_bbox: Optional[np.ndarray] = None  # None = no clear single subject
    primary_subject_type: str = "face"                 # face|object|scene
    vl_description:       str = ""
    suggested_layout:     str = "single_crop"


# ── VLM prompt ────────────────────────────────────────────────────────────────

_VL_PROMPT = """You are a short-form video editor. Analyse this frame for 9:16 reframing.

Reply in JSON only:
{
  "layout_type": "<single_person|two_people|three_people|four_people|gaming_facecam|screen_share|reaction_video|scenic|product_demo|cooking|performance|duet>",
  "primary_subject": "<brief description>",
  "primary_bbox": [x1_norm, y1_norm, x2_norm, y2_norm],
  "notes": "<any framing notes>"
}"""

_VL_LAYOUT_MAP = {
    "single_person":  "single_crop",
    "two_people":     "split_v_2",
    "three_people":   "split_v_3",
    "four_people":    "grid_2x2",
    "gaming_facecam": "pip",
    "screen_share":   "screen_person",
    "reaction_video": "reaction",
    "scenic":         "saliency",
    "product_demo":   "screen_person",
    "cooking":        "single_crop",
    "performance":    "single_crop",
    "duet":           "horizontal_2",
}


# ── Detection helpers ─────────────────────────────────────────────────────────

def _detect_faces(frame: np.ndarray, face_app) -> list[FaceResult]:
    try:
        raw = face_app.get(frame)
    except Exception as exc:
        log.warning("InsightFace detection error: %s", exc)
        return []

    src_h, src_w = frame.shape[:2]
    cx0, cy0 = src_w / 2.0, src_h / 2.0
    results = []

    for f in raw:
        bbox = f.bbox.astype(float)
        cx   = (bbox[0] + bbox[2]) / 2.0
        cy   = (bbox[1] + bbox[3]) / 2.0
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        dist = np.hypot((cx - cx0) / src_w, (cy - cy0) / src_h)
        emb  = f.embedding
        emb  = emb / (np.linalg.norm(emb) + 1e-8)
        results.append(FaceResult(bbox=bbox, embedding=emb, cx=cx, cy=cy,
                                  area=area, center_dist=dist))

    results.sort(key=lambda r: r.area, reverse=True)
    return results


def _detect_objects(frame: np.ndarray, gdino_model, prompt: str) -> list[ObjectResult]:
    try:
        from groundingdino.util.inference import predict as gdino_predict
        from PIL import Image

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        boxes, logits, phrases = gdino_predict(
            model=gdino_model,
            image=pil_img,
            caption=prompt,
            box_threshold=0.35,
            text_threshold=0.25,
        )

        h, w = frame.shape[:2]
        results = []
        for box, score, label in zip(boxes, logits, phrases):
            cx_n, cy_n, bw_n, bh_n = box.tolist()
            x1 = (cx_n - bw_n / 2) * w
            y1 = (cy_n - bh_n / 2) * h
            x2 = (cx_n + bw_n / 2) * w
            y2 = (cy_n + bh_n / 2) * h
            results.append(ObjectResult(
                bbox=np.array([x1, y1, x2, y2]),
                label=label, score=float(score),
                cx=cx_n * w, cy=cy_n * h,
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results
    except Exception as exc:
        log.warning("GroundingDINO detection error: %s", exc)
        return []


def _query_vl(frame: np.ndarray, vl_model, vl_processor) -> str:
    try:
        import torch
        from PIL import Image

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text",  "text": _VL_PROMPT},
        ]}]

        text   = vl_processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        inputs = vl_processor(text=[text], images=[pil_img],
                               padding=True, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            out_ids = vl_model.generate(**inputs, max_new_tokens=256)

        return vl_processor.batch_decode(
            out_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0].strip()
    except Exception as exc:
        log.warning("Qwen2-VL query failed: %s", exc)
        return ""


def _parse_vl_response(text: str, src_w: int, src_h: int) -> tuple[str, Optional[np.ndarray]]:
    """Extract layout suggestion and optional primary bbox from VLM JSON response."""
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return "single_crop", None
        data     = json.loads(m.group())
        layout   = _VL_LAYOUT_MAP.get(data.get("layout_type", ""), "single_crop")
        bbox_raw = data.get("primary_bbox")
        bbox = None
        if bbox_raw and len(bbox_raw) == 4:
            x1, y1, x2, y2 = bbox_raw
            bbox = np.array([x1 * src_w, y1 * src_h, x2 * src_w, y2 * src_h])
        return layout, bbox
    except Exception:
        return "single_crop", None


# ── Public entry point ────────────────────────────────────────────────────────

def analyze_shot_frame(
    frame:        np.ndarray,
    category:     str,
    gdino_prompt: str,
    face_app=None,
    gdino_model=None,
    vl_model=None,
    vl_processor=None,
) -> ShotAnalysisResult:
    """
    Analyse the first frame of a shot.  Model arguments are all optional —
    the function degrades gracefully when any model is unavailable.
    """
    src_h, src_w = frame.shape[:2]

    # ── 1. Face detection (always fast) ───────────────────────────────────────
    faces = _detect_faces(frame, face_app) if face_app else []
    n     = len(faces)

    # ── 2. Object detection for non-face-primary categories ──────────────────
    needs_objects = (
        n == 0
        or category in ("gaming", "cooking_food", "sports_action",
                        "product_review", "keynote_talk", "reaction_video",
                        "asmr_lofi", "travel_scenery")
    )
    objects: list[ObjectResult] = []
    if needs_objects and gdino_model:
        objects = _detect_objects(frame, gdino_model, gdino_prompt)

    # ── 3. Classify shot type and select primary subject ─────────────────────
    vl_desc       = ""
    suggested     = "single_crop"
    primary_bbox: Optional[np.ndarray] = None
    primary_type  = "face"

    if n == 0 and not objects:
        shot_type    = "scenic"
        primary_type = "scene"
        suggested    = "saliency"

    elif n == 0:
        shot_type    = "object"
        primary_type = "object"
        primary_bbox = objects[0].bbox

        if category == "gaming":
            suggested = "pip" if any("screen" in o.label for o in objects) else "screen_person"
        elif category == "keynote_talk":
            suggested = "screen_person"
        elif category == "reaction_video":
            suggested = "reaction"
        else:
            suggested = "single_crop"

    elif n == 1:
        shot_type    = "solo"
        primary_bbox = faces[0].bbox

        if category == "product_review" and objects:
            suggested = "screen_person"   # face top, product bottom
        elif category == "reaction_video":
            # Ask VLM whether there is source content in the frame
            if vl_model:
                vl_desc   = _query_vl(frame, vl_model, vl_processor)
                vl_layout, vl_bbox = _parse_vl_response(vl_desc, src_w, src_h)
                suggested = vl_layout
            else:
                suggested = "single_crop"
        elif category == "gaming":
            suggested = "pip"
        else:
            suggested = "single_crop"

    elif n == 2:
        shot_type = "dual"
        primary_bbox = None  # both faces are principal

        if category == "gaming":
            suggested = "pip"
        elif category == "reaction_video":
            suggested = "reaction"
        else:
            suggested = "split_v_2"

    else:
        # 3+ faces — call VLM to find the lead and suggest layout
        shot_type = "group"

        if vl_model:
            vl_desc   = _query_vl(frame, vl_model, vl_processor)
            vl_layout, vl_bbox = _parse_vl_response(vl_desc, src_w, src_h)
            suggested    = vl_layout
            primary_bbox = vl_bbox

        if primary_bbox is None:
            # Fallback: face with most screen time proxy = largest + most centred
            best = min(faces, key=lambda f: f.center_dist + (1.0 - f.area / (src_w * src_h)))
            primary_bbox = best.bbox

        if not suggested or suggested == "single_crop":
            suggested = {3: "split_v_3", 4: "grid_2x2"}.get(n, "single_crop")

    log.debug(
        "analyze_shot: type=%s faces=%d objects=%d layout=%s vl=%s",
        shot_type, n, len(objects), suggested, bool(vl_desc),
    )

    return ShotAnalysisResult(
        shot_type=shot_type,
        faces=faces,
        objects=objects,
        primary_subject_bbox=primary_bbox,
        primary_subject_type=primary_type,
        vl_description=vl_desc,
        suggested_layout=suggested,
    )
