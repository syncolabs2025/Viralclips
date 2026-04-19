"""
Layout compositor for 9:16 output frames.

A Layout is a set of named Slots, each with a normalised position/size in the
1080×1920 output canvas.  compose_frame() crops the source frame for every
slot and assembles the final canvas in one pass.
"""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

OUT_W = 1080
OUT_H = 1920


class LayoutType(str, Enum):
    SINGLE_CROP   = "single_crop"    # 1 subject fills the full frame
    SPLIT_V_2     = "split_v_2"      # 2-person: active speaker 60% top, listener 40% bottom
    SPLIT_V_3     = "split_v_3"      # 3-person: two side-by-side top, speaker full-width bottom
    GRID_2x2      = "grid_2x2"       # 4-person 2×2 grid
    PIP           = "pip"            # main content fullscreen + small face-cam overlay
    SCREEN_PERSON = "screen_person"  # screen/game top 62%, presenter bottom 38%
    REACTION      = "reaction"       # source content top 50%, reactor face bottom 50%
    SCREEN_3SLOT  = "screen_3slot"   # person A top, shared screen middle, person B bottom
    HORIZONTAL_2  = "horizontal_2"   # two performers stacked equally (music/duet)
    SALIENCY      = "saliency"       # no fixed subject — saliency centroid drives crop


@dataclass(frozen=True)
class SlotDef:
    """Normalised slot rectangle [0, 1] within the 9:16 output canvas."""
    role: str
    x:    float   # left edge
    y:    float   # top edge
    w:    float   # width
    h:    float   # height


# ── Layout tables ─────────────────────────────────────────────────────────────
#
# Roles used as keys in slot_centres dicts throughout the engine:
#   active / passive        — podcast / interview speaker slots
#   speaker                 — current speaker in 3-person layout (bottom slot)
#   active_a / active_b     — left / right in 3-person top row
#   person_0..3             — grid layout slots
#   main                    — primary subject in single / PiP / scenic layouts
#   overlay                 — PiP face-cam overlay
#   screen                  — screen / slide / game content region
#   source                  — original video being reacted to
#   reactor                 — person reacting
#   performer_a / _b        — two performers in horizontal duet layout
#   scene / saliency        — saliency-driven crop (no fixed subject)

LAYOUT_SLOTS: dict[LayoutType, list[SlotDef]] = {
    LayoutType.SINGLE_CROP: [
        SlotDef("main",        0.00, 0.00, 1.00, 1.00),
    ],
    LayoutType.SPLIT_V_2: [
        SlotDef("active",      0.00, 0.00, 1.00, 0.60),
        SlotDef("passive",     0.00, 0.60, 1.00, 0.40),
    ],
    LayoutType.SPLIT_V_3: [
        SlotDef("active_a",    0.00, 0.00, 0.50, 0.55),   # top-left
        SlotDef("active_b",    0.50, 0.00, 0.50, 0.55),   # top-right
        SlotDef("speaker",     0.00, 0.55, 1.00, 0.45),   # bottom full — current talker
    ],
    LayoutType.GRID_2x2: [
        SlotDef("person_0",    0.00, 0.00, 0.50, 0.50),
        SlotDef("person_1",    0.50, 0.00, 0.50, 0.50),
        SlotDef("person_2",    0.00, 0.50, 0.50, 0.50),
        SlotDef("person_3",    0.50, 0.50, 0.50, 0.50),
    ],
    LayoutType.PIP: [
        SlotDef("main",        0.00, 0.00, 1.00, 1.00),
        SlotDef("overlay",     0.60, 0.72, 0.38, 0.26),   # bottom-right face cam
    ],
    LayoutType.SCREEN_PERSON: [
        SlotDef("screen",      0.00, 0.00, 1.00, 0.62),
        SlotDef("person",      0.00, 0.62, 1.00, 0.38),
    ],
    LayoutType.REACTION: [
        SlotDef("source",      0.00, 0.00, 1.00, 0.50),
        SlotDef("reactor",     0.00, 0.50, 1.00, 0.50),
    ],
    LayoutType.SCREEN_3SLOT: [
        SlotDef("person_a",    0.00, 0.00, 1.00, 0.27),
        SlotDef("screen",      0.00, 0.27, 1.00, 0.46),
        SlotDef("person_b",    0.00, 0.73, 1.00, 0.27),
    ],
    LayoutType.HORIZONTAL_2: [
        SlotDef("performer_a", 0.00, 0.00, 1.00, 0.50),
        SlotDef("performer_b", 0.00, 0.50, 1.00, 0.50),
    ],
    LayoutType.SALIENCY: [
        SlotDef("scene",       0.00, 0.00, 1.00, 1.00),
    ],
}

# Roles that get headroom applied (face/body subjects — shift crop up slightly)
_HEADROOM_ROLES = frozenset({
    "active", "passive", "speaker", "active_a", "active_b",
    "main", "reactor", "person", "person_a", "person_b",
    "person_0", "person_1", "person_2", "person_3",
    "performer_a", "performer_b",
})


def _slot_px(slot: SlotDef) -> tuple[int, int, int, int]:
    """Normalised SlotDef → pixel (x1, y1, w, h) in the output canvas."""
    return (
        int(slot.x * OUT_W),
        int(slot.y * OUT_H),
        max(1, int(slot.w * OUT_W)),
        max(1, int(slot.h * OUT_H)),
    )


def crop_to_slot(
    frame:         np.ndarray,
    cx:            float,
    cy:            float,
    slot_w:        int,
    slot_h:        int,
    headroom:      float = 0.0,
) -> np.ndarray:
    """
    Crop a slot_w×slot_h region from frame centred on (cx, cy), apply
    headroom shift, then resize to exactly slot_w×slot_h.
    """
    src_h, src_w = frame.shape[:2]
    aspect = slot_w / slot_h

    crop_h = min(src_h, int(src_w / aspect))
    crop_w = int(crop_h * aspect)
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(crop_w / aspect)

    # Headroom: shift the crop window upward so face isn't dead-centre
    cy_adj = cy - headroom * crop_h

    x1 = int(cx - crop_w / 2)
    y1 = int(cy_adj - crop_h / 2)
    x1 = max(0, min(x1, src_w - crop_w))
    y1 = max(0, min(y1, src_h - crop_h))

    cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(cropped, (slot_w, slot_h), interpolation=cv2.INTER_LINEAR)


def compose_frame(
    source:       np.ndarray,
    layout:       LayoutType,
    slot_centres: dict[str, tuple[float, float]],
    headroom:     float = 0.15,
) -> np.ndarray:
    """
    Assemble one 1080×1920 output frame.

    slot_centres maps slot role → (cx, cy) in *source* pixel coords.
    Missing roles fall back to the source frame centre.
    """
    canvas  = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
    src_h, src_w = source.shape[:2]
    default = (src_w / 2.0, src_h / 2.0)

    for slot in LAYOUT_SLOTS[layout]:
        x1, y1, sw, sh = _slot_px(slot)
        cx, cy = slot_centres.get(slot.role, default)
        hr = headroom if slot.role in _HEADROOM_ROLES else 0.0
        patch = crop_to_slot(source, cx, cy, sw, sh, hr)
        canvas[y1 : y1 + sh, x1 : x1 + sw] = patch

    # White border around PiP overlay for visual separation
    if layout == LayoutType.PIP:
        pip = next(s for s in LAYOUT_SLOTS[layout] if s.role == "overlay")
        px1, py1, pw, ph = _slot_px(pip)
        cv2.rectangle(canvas, (px1 - 3, py1 - 3), (px1 + pw + 3, py1 + ph + 3),
                      (255, 255, 255), 3)

    # Thin divider line between stacked slots to make the split read clearly
    if layout in (LayoutType.SPLIT_V_2, LayoutType.SCREEN_PERSON, LayoutType.REACTION):
        split_y = int(LAYOUT_SLOTS[layout][0].h * OUT_H)
        cv2.line(canvas, (0, split_y), (OUT_W, split_y), (30, 30, 30), 2)

    return canvas
