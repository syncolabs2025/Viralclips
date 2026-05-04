"""
ReframeEngine — main orchestrator.

Pipeline per clip:
  1. Shot detection        (TransNetV2 + PySceneDetect union)
  2. Audio analysis        (pyannote diarization + librosa beats)
  3. Per-shot first-frame  (InsightFace + GroundingDINO + optional Qwen2-VL)
  4. Cross-shot identity   (ArcFace DBSCAN clustering → stable person IDs)
  5. Per-shot tracking     (SAM-2 with Kalman fallback, hard-reset per shot)
  6. Render                (layout compositor → FFmpeg NVENC pipe per shot)
  7. Concatenate + mux     (FFmpeg concat + audio mux)

All models are injected at construction time — the engine itself is stateless
with respect to model weights and can process many clips in sequence.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .audio_analysis import AudioAnalysis, analyze_audio, active_speaker_at, nearest_beat
from .categories import ContentCategory, CATEGORY_RULES, ReframeRule
from .identity import cluster_identities, identify_lead_performer
from .layouts import LayoutType, LAYOUT_SLOTS, OUT_W, OUT_H, compose_frame
from .shot_analysis import ShotAnalysisResult, analyze_shot_frame
from .shot_detection import Shot, detect_shots
from .tracking import TrackResult, track_with_sam2, track_with_kalman, track_saliency

log = logging.getLogger("reframe.engine")


# ── FFmpeg helpers ────────────────────────────────────────────────────────────

def _has_nvenc() -> bool:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return "h264_nvenc" in out
    except Exception:
        return False


_ENCODER = "h264_nvenc" if _has_nvenc() else "libx264"
log.info("Video encoder: %s", _ENCODER)


def _open_pipe(path: str, fps: float) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{OUT_W}x{OUT_H}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "pipe:0", "-vcodec", _ENCODER,
    ]
    if _ENCODER == "h264_nvenc":
        cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "24"]
    else:
        cmd += ["-preset", "fast", "-crf", "22"]
    cmd += ["-pix_fmt", "yuv420p", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _mux_audio(vid_only: str, source: str, start: float, duration: float, out: str) -> None:
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start), "-t", str(duration), "-i", source,
        "-i", vid_only,
        "-map", "1:v", "-map", "0:a",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", out,
    ], check=True, capture_output=True)


def _concat_shots(shot_files: list[str], work_dir: Path, out: str) -> None:
    lst = str(work_dir / "concat.txt")
    with open(lst, "w") as f:
        for p in shot_files:
            f.write(f"file '{p}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out],
        check=True, capture_output=True,
    )


def _add_watermark(frame: np.ndarray) -> np.ndarray:
    h, w   = frame.shape[:2]
    text   = "ViralClips"
    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale  = w / 800
    thick  = max(1, int(scale * 2))
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    x = (w - tw) // 2
    y = h - 28
    cv2.putText(frame, text, (x + 2, y + 2), font, scale, (0, 0, 0),     thick + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x,     y    ), font, scale, (220, 220, 220), thick,     cv2.LINE_AA)
    return frame


def _get_frame(video_path: str, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok or frame is None:
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
        frame = np.zeros((h, w, 3), dtype=np.uint8)
    cap.release()
    return frame


# ── Layout mapping ────────────────────────────────────────────────────────────

_LAYOUT_STR_MAP: dict[str, LayoutType] = {lt.value: lt for lt in LayoutType}


# ── Per-shot render plan ──────────────────────────────────────────────────────

@dataclass
class ShotPlan:
    shot:     Shot
    layout:   LayoutType
    tracks:   dict[str, TrackResult]      # role → track
    analysis: ShotAnalysisResult


# ── Engine ────────────────────────────────────────────────────────────────────

class ReframeEngine:
    """
    Inject loaded models at construction; call process() for each clip.
    All model references are optional — every layer degrades gracefully.
    """

    def __init__(
        self,
        face_app=None,       # insightface.app.FaceAnalysis (buffalo_l, CUDA)
        sam2=None,           # sam2.build_sam.build_sam2_video_predictor result
        vl_model=None,       # Qwen2-VL model (transformers, INT8)
        vl_processor=None,   # Qwen2-VL processor
        gdino=None,          # GroundingDINO model
        work_dir: Optional[Path] = None,
    ):
        self.face_app     = face_app
        self.sam2         = sam2
        self.vl_model     = vl_model
        self.vl_processor = vl_processor
        self.gdino        = gdino
        self.work_dir     = work_dir or Path(tempfile.mkdtemp(prefix="reframe_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────────────

    def process(
        self,
        video_path:  str,
        start:       float,
        end:         float,
        category:    ContentCategory,
        output_path: str,
        watermark:   bool = False,
    ) -> None:
        """
        Full per-clip reframe pipeline.  Writes the final 1080×1920 MP4 to
        output_path with audio muxed from the source.
        """
        rule     = CATEGORY_RULES[category]
        duration = end - start
        log.info("Reframing [%.1fs–%.1fs] category=%s", start, end, category.value)

        # 1 ── Shot detection ──────────────────────────────────────────────────
        shots = detect_shots(video_path, start, end)

        # 2 ── Audio analysis ─────────────────────────────────────────────────
        audio = analyze_audio(video_path, is_music=(category == ContentCategory.MUSIC_VIDEO))

        # 3 ── Per-shot first-frame analysis ──────────────────────────────────
        analyses: list[ShotAnalysisResult] = []
        for shot in shots:
            frame    = _get_frame(video_path, shot.start_time + 0.05)
            analysis = analyze_shot_frame(
                frame=frame,
                category=category.value,
                gdino_prompt=rule.gdino_prompt,
                face_app=self.face_app,
                gdino_model=self.gdino,
                vl_model=self.vl_model,
                vl_processor=self.vl_processor,
            )
            analyses.append(analysis)
            log.info("  Shot %d [%.1f–%.1f]: %s → %s  faces=%d",
                     shot.index, shot.start_time, shot.end_time,
                     analysis.shot_type, analysis.suggested_layout, len(analysis.faces))

        # 4 ── Cross-shot identity clustering ─────────────────────────────────
        shot_embeddings = [[f.embedding for f in a.faces] for a in analyses]
        shot_durations  = [s.duration for s in shots]
        shot_to_persons = cluster_identities(shot_embeddings, shot_durations)
        lead_id         = identify_lead_performer(shot_to_persons, analyses, shot_durations)

        # 5 ── Per-shot tracking ───────────────────────────────────────────────
        plans: list[ShotPlan] = []
        for shot, analysis in zip(shots, analyses):
            layout = self._resolve_layout(shot, analysis, category, audio, shot_to_persons)
            tracks = self._build_tracks(
                video_path, shot, analysis, layout, rule,
                shot_to_persons, lead_id, audio,
            )
            plans.append(ShotPlan(shot, layout, tracks, analysis))

        # 6 ── Render each shot ────────────────────────────────────────────────
        shot_files: list[str] = []
        for plan in plans:
            out = str(self.work_dir / f"shot_{plan.shot.index:04d}.mp4")
            self._render_shot(video_path, plan, rule, out, watermark)
            shot_files.append(out)

        # 7 ── Concat shots + mux audio ────────────────────────────────────────
        vid_only = str(self.work_dir / "vidonly.mp4")
        _concat_shots(shot_files, self.work_dir, vid_only)
        _mux_audio(vid_only, video_path, start, duration, output_path)

        for f in shot_files + [vid_only]:
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass

        log.info("Done → %s", output_path)

    # ── Layout resolution ─────────────────────────────────────────────────────

    def _resolve_layout(
        self,
        shot:           Shot,
        analysis:       ShotAnalysisResult,
        category:       ContentCategory,
        audio:          AudioAnalysis,
        shot_to_persons: dict,
    ) -> LayoutType:
        """
        Map shot analysis + category → LayoutType.
        Category-level rules can override what the shot analysis suggested.
        """
        base = _LAYOUT_STR_MAP.get(analysis.suggested_layout, LayoutType.SINGLE_CROP)

        # Hard category overrides
        if category == ContentCategory.GAMING:
            if analysis.shot_type in ("solo", "dual", "group"):
                return LayoutType.PIP            # game full + face-cam corner
            return LayoutType.SCREEN_PERSON      # game top, no face bottom

        if category == ContentCategory.REACTION_VIDEO:
            if analysis.shot_type != "scenic":
                return LayoutType.REACTION

        if category in (ContentCategory.FITNESS, ContentCategory.MUSIC_VIDEO):
            if analysis.shot_type == "solo":
                return LayoutType.SINGLE_CROP    # full-body; SAM-2 handles bbox expansion

        if category == ContentCategory.KEYNOTE_TALK:
            if analysis.objects:
                return LayoutType.SCREEN_PERSON

        return base

    # ── Track builder ─────────────────────────────────────────────────────────

    def _build_tracks(
        self,
        video_path:     str,
        shot:           Shot,
        analysis:       ShotAnalysisResult,
        layout:         LayoutType,
        rule:           ReframeRule,
        shot_to_persons: dict[int, list[str]],
        lead_id:        Optional[str],
        audio:          AudioAnalysis,
    ) -> dict[str, TrackResult]:
        """Return {slot_role: TrackResult} for every slot in the layout."""
        tracks: dict[str, TrackResult] = {}
        faces   = analysis.faces
        objects = analysis.objects
        n       = len(faces)

        for slot in LAYOUT_SLOTS[layout]:
            role = slot.role
            tracks[role] = self._track_for_role(
                role, video_path, shot, analysis, rule,
                shot_to_persons, lead_id, audio, faces, objects, n,
            )

        return tracks

    def _track_for_role(
        self, role, video_path, shot, analysis, rule,
        shot_to_persons, lead_id, audio, faces, objects, n,
    ) -> TrackResult:
        """Dispatch tracking for a single slot role."""

        # ── Scene / saliency roles ────────────────────────────────────────────
        if role in ("scene", "saliency"):
            return track_saliency(video_path, shot.start_time, shot.end_time,
                                  shot.fps, ema_alpha=rule.ema_alpha)

        # ── Screen / source content ───────────────────────────────────────────
        if role in ("screen", "source", "main") and not faces:
            if objects:
                return self._track(video_path, shot, objects[0].bbox, rule)
            return track_saliency(video_path, shot.start_time, shot.end_time,
                                  shot.fps, ema_alpha=rule.ema_alpha)

        # ── Full-frame main (single subject) ─────────────────────────────────
        if role == "main":
            bbox = analysis.primary_subject_bbox
            if bbox is None and faces:
                bbox = self._lead_bbox(faces, shot_to_persons, shot.index, lead_id)
            return self._track(video_path, shot, bbox, rule)

        # ── Podcast / interview speaker slots ─────────────────────────────────
        if role == "active":
            face = self._pick_speaker(faces, shot, audio, shot_to_persons, lead_id, prefer_lead=True)
            return self._track(video_path, shot, face.bbox if face else None, rule)

        if role == "passive":
            speaker = self._pick_speaker(faces, shot, audio, shot_to_persons, lead_id, prefer_lead=True)
            passive = next((f for f in faces if f is not speaker), None) if speaker else None
            if passive is None and faces:
                passive = faces[-1]
            return self._track(video_path, shot, passive.bbox if passive else None, rule)

        # ── 3-person bottom speaker slot ──────────────────────────────────────
        if role == "speaker":
            face = self._pick_speaker(faces, shot, audio, shot_to_persons, lead_id, prefer_lead=False)
            return self._track(video_path, shot, face.bbox if face else None, rule)

        # ── 3-person top row ──────────────────────────────────────────────────
        if role == "active_a":
            return self._track(video_path, shot, faces[0].bbox if n >= 1 else None, rule)
        if role == "active_b":
            return self._track(video_path, shot, faces[1].bbox if n >= 2 else None, rule)

        # ── 4-person grid ─────────────────────────────────────────────────────
        if role.startswith("person_") and role[7:].isdigit():
            idx = int(role[7:])
            bbox = faces[idx].bbox if idx < n else (faces[0].bbox if faces else None)
            return self._track(video_path, shot, bbox, rule)

        # ── PiP overlay (face cam) ─────────────────────────────────────────────
        if role == "overlay":
            # Smallest face = the face cam (the game is the large region)
            face = min(faces, key=lambda f: f.area) if faces else None
            return self._track(video_path, shot, face.bbox if face else None, rule)

        # ── Reaction reactor ─────────────────────────────────────────────────
        if role == "reactor":
            face = self._pick_speaker(faces, shot, audio, shot_to_persons, lead_id, prefer_lead=True)
            return self._track(video_path, shot, face.bbox if face else None, rule)

        # ── Person slot in SCREEN_PERSON / SCREEN_3SLOT ───────────────────────
        if role == "person":
            face = self._lead_face(faces, shot_to_persons, shot.index, lead_id)
            return self._track(video_path, shot, face.bbox if face else None, rule)

        if role == "person_a":
            return self._track(video_path, shot, faces[0].bbox if n >= 1 else None, rule)
        if role == "person_b":
            return self._track(video_path, shot, faces[-1].bbox if n >= 2 else None, rule)

        # ── Music / duet performers ───────────────────────────────────────────
        if role == "performer_a":
            return self._track(video_path, shot, faces[0].bbox if n >= 1 else None, rule)
        if role == "performer_b":
            return self._track(video_path, shot, faces[1].bbox if n >= 2 else None, rule)

        # ── Final fallback ────────────────────────────────────────────────────
        return track_saliency(video_path, shot.start_time, shot.end_time, shot.fps)

    def _track(
        self,
        video_path: str,
        shot:       Shot,
        bbox:       Optional[np.ndarray],
        rule:       ReframeRule,
    ) -> TrackResult:
        """Route to SAM-2 → Kalman depending on availability."""
        if self.sam2 is not None and bbox is not None:
            return track_with_sam2(video_path, shot.start_time, shot.end_time,
                                   bbox, self.sam2, shot.fps)
        return track_with_kalman(
            video_path, shot.start_time, shot.end_time, bbox, shot.fps,
            process_var=rule.kalman_process_var,
            measure_var=rule.kalman_measure_var,
            detection_fps=rule.detection_fps,
        )

    # ── Speaker / lead helpers ────────────────────────────────────────────────

    def _pick_speaker(self, faces, shot, audio, shot_to_persons, lead_id, prefer_lead):
        """Return the face that is currently speaking via pyannote, or fallback."""
        if not faces:
            return None
        if not audio.speaker_segments:
            return faces[0] if prefer_lead else faces[-1]

        mid = (shot.start_time + shot.end_time) / 2.0
        label = active_speaker_at(mid, audio.speaker_segments)
        if label is None:
            return faces[0]

        person_ids = shot_to_persons.get(shot.index, [])
        for i, pid in enumerate(person_ids):
            if label in pid and i < len(faces):
                return faces[i]
        return faces[0]

    def _lead_face(self, faces, shot_to_persons, shot_index, lead_id):
        if not faces:
            return None
        if lead_id is None:
            return faces[0]
        pids = shot_to_persons.get(shot_index, [])
        for i, pid in enumerate(pids):
            if pid == lead_id and i < len(faces):
                return faces[i]
        return faces[0]

    def _lead_bbox(self, faces, shot_to_persons, shot_index, lead_id):
        f = self._lead_face(faces, shot_to_persons, shot_index, lead_id)
        return f.bbox if f else None

    # ── Shot renderer ─────────────────────────────────────────────────────────

    def _render_shot(
        self,
        video_path: str,
        plan:       ShotPlan,
        rule:       ReframeRule,
        out_path:   str,
        watermark:  bool,
    ) -> None:
        """Render one shot to out_path (video only, no audio)."""
        cap     = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        total = plan.shot.frame_count
        proc  = _open_pipe(out_path, src_fps)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, plan.shot.start_time * 1000)

        default = (src_w / 2.0, src_h / 2.0)

        for i in range(total):
            ok, frame = cap.read()
            if not ok:
                break

            slot_centres: dict[str, tuple[float, float]] = {
                role: (track.centres[i] if i < len(track.centres) else default)
                for role, track in plan.tracks.items()
            }

            out = compose_frame(frame, plan.layout, slot_centres, rule.headroom_ratio)
            if watermark:
                out = _add_watermark(out)
            proc.stdin.write(out.tobytes())

        proc.stdin.close()
        proc.wait()
        cap.release()

        log.debug("Shot %d rendered → %s", plan.shot.index, out_path)
