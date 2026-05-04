"""
Subject tracking within a single shot.

Primary:  SAM-2 Large  — pixel-perfect mask propagation.
          Frames are extracted to a temp directory, SAM-2 is initialised with
          the subject bbox on frame 0, then propagated through the shot.
          The tracker is hard-reset between shots.

Fallback: Kalman-filtered MediaPipe face detection at 5 FPS.
          Applied when SAM-2 is unavailable or raises an exception.

Both return a TrackResult with one (cx, cy) centre per source frame,
ready to be consumed by the layout compositor.
"""

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("reframe.tracking")


# ── Kalman filter ─────────────────────────────────────────────────────────────

class _Kalman1D:
    """Minimal 1-D Kalman filter (position + velocity state)."""

    def __init__(self, process_var: float = 5.0, measure_var: float = 200.0):
        self._x: Optional[float] = None
        self._v  = 0.0
        self._p  = 1e4
        self._q  = process_var
        self._r  = measure_var

    def reset(self) -> None:
        self._x = None
        self._v = 0.0
        self._p = 1e4

    def update(self, z: float) -> float:
        if self._x is None:
            self._x = z
            return z
        x_p = self._x + self._v
        p_p = self._p + self._q
        k      = p_p / (p_p + self._r)
        self._x = x_p + k * (z - x_p)
        self._v += 0.3 * (self._x - x_p)
        self._p = (1 - k) * p_p
        return self._x


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class TrackResult:
    centres: list[tuple[float, float]]                  # (cx, cy) per frame in shot


# ── SAM-2 tracking ────────────────────────────────────────────────────────────

def _extract_frames(video_path: str, start: float, end: float, fps: float) -> tuple[str, int]:
    """Write JPEG frames for [start, end] to a temp dir.  Returns (dir, n_frames)."""
    tmpdir = tempfile.mkdtemp(prefix="sam2_")
    cap    = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    total   = int((end - start) * src_fps)
    n       = 0
    while n < total:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(tmpdir, f"{n:06d}.jpg"), frame)
        n += 1
    cap.release()
    return tmpdir, n


def _mask_centroid(mask: np.ndarray) -> Optional[tuple[float, float]]:
    ys, xs = np.where(mask > 0.5)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def track_with_sam2(
    video_path:     str,
    shot_start:     float,
    shot_end:       float,
    subject_bbox:   np.ndarray,
    sam2_predictor,
    fps:            float,
) -> TrackResult:
    """
    SAM-2 video predictor.  Re-initialised fresh each call (shot boundary reset).
    Falls back to Kalman on any exception.
    """
    try:
        import torch

        frames_dir, n_frames = _extract_frames(video_path, shot_start, shot_end, fps)
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = sam2_predictor.init_state(video_path=frames_dir)

                x1, y1, x2, y2 = subject_bbox.tolist()
                sam2_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    box=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                )

                raw: dict[int, tuple[float, float]] = {}
                for fidx, _, logits in sam2_predictor.propagate_in_video(state):
                    mask = (logits[0] > 0.0).squeeze().cpu().numpy()
                    c    = _mask_centroid(mask)
                    if c is not None and fidx < n_frames:
                        raw[fidx] = c

                sam2_predictor.reset_state(state)
        finally:
            shutil.rmtree(frames_dir, ignore_errors=True)

        # Fill gaps, build per-frame list
        cap   = cv2.VideoCapture(video_path)
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        fallback = (src_w / 2.0, src_h / 2.0)

        centres: list[tuple[float, float]] = []
        last = fallback
        for i in range(n_frames):
            if i in raw:
                last = raw[i]
            centres.append(last)

        log.debug("SAM-2 tracked %d/%d frames for shot [%.1f–%.1f]",
                  len(raw), n_frames, shot_start, shot_end)
        return TrackResult(centres=centres)

    except Exception as exc:
        log.warning("SAM-2 failed (%s) — falling back to Kalman", exc)
        return track_with_kalman(video_path, shot_start, shot_end, subject_bbox, fps)


# ── Kalman tracking ───────────────────────────────────────────────────────────

def track_with_kalman(
    video_path:     str,
    shot_start:     float,
    shot_end:       float,
    subject_bbox:   Optional[np.ndarray],
    fps:            float,
    process_var:    float = 5.0,
    measure_var:    float = 200.0,
    detection_fps:  float = 5.0,
) -> TrackResult:
    """
    MediaPipe face detection at detection_fps + Kalman smoothing + linear
    interpolation to every frame.  Always re-initialised from subject_bbox.
    """
    import mediapipe as mp
    _mp_face = mp.solutions.face_detection

    cap     = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    total = int((shot_end - shot_start) * src_fps)
    step  = max(1, int(src_fps / detection_fps))

    init_cx = (subject_bbox[0] + subject_bbox[2]) / 2.0 if subject_bbox is not None else src_w / 2.0
    init_cy = (subject_bbox[1] + subject_bbox[3]) / 2.0 if subject_bbox is not None else src_h / 2.0

    raw_cx: dict[int, float] = {0: init_cx}
    raw_cy: dict[int, float] = {0: init_cy}

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, shot_start * 1000)

    with _mp_face.FaceDetection(min_detection_confidence=0.45) as det:
        for i in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            if i % step != 0:
                continue
            result = det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.detections:
                best = max(
                    result.detections,
                    key=lambda d: (d.location_data.relative_bounding_box.width
                                   * d.location_data.relative_bounding_box.height),
                )
                bb = best.location_data.relative_bounding_box
                raw_cx[i] = (bb.xmin + bb.width  / 2) * src_w
                raw_cy[i] = (bb.ymin + bb.height / 2) * src_h
    cap.release()

    # Fill gaps → Kalman smooth → interpolate to every frame
    last_x, last_y = init_cx, init_cy
    filled_cx: dict[int, float] = {}
    filled_cy: dict[int, float] = {}
    for i in range(total):
        if i in raw_cx:
            last_x, last_y = raw_cx[i], raw_cy[i]
        filled_cx[i] = last_x
        filled_cy[i] = last_y

    kx = _Kalman1D(process_var, measure_var)
    ky = _Kalman1D(process_var, measure_var)
    smoothed: dict[int, tuple[float, float]] = {}
    for i in sorted(filled_cx):
        smoothed[i] = (kx.update(filled_cx[i]), ky.update(filled_cy[i]))

    keys = sorted(smoothed)
    centres: list[tuple[float, float]] = []
    for i in range(total):
        lo = max((k for k in keys if k <= i), default=keys[0])
        hi = min((k for k in keys if k >= i), default=keys[-1])
        if lo == hi:
            centres.append(smoothed[lo])
        else:
            t  = (i - lo) / (hi - lo)
            cx = smoothed[lo][0] + t * (smoothed[hi][0] - smoothed[lo][0])
            cy = smoothed[lo][1] + t * (smoothed[hi][1] - smoothed[lo][1])
            centres.append((cx, cy))

    return TrackResult(centres=centres)


# ── Saliency tracking ─────────────────────────────────────────────────────────

def track_saliency(
    video_path:  str,
    shot_start:  float,
    shot_end:    float,
    fps:         float,
    ema_alpha:   float = 0.08,
    sample_fps:  float = 2.0,
) -> TrackResult:
    """
    SpectralResidual saliency centroid with EMA smoothing.
    Used for scenic / no-subject shots.
    """
    cap     = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    total = int((shot_end - shot_start) * src_fps)
    step  = max(1, int(src_fps / sample_fps))
    det   = cv2.saliency.StaticSaliencySpectralResidual_create()

    raw: dict[int, tuple[float, float]] = {}
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, shot_start * 1000)
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if i % step != 0:
            continue
        ok2, smap = det.computeSaliency(frame)
        if not ok2:
            raw[i] = (src_w / 2.0, src_h / 2.0)
            continue
        smap   = smap.squeeze().astype(np.float32)
        thresh = np.percentile(smap, 60)
        mask   = (smap >= thresh).astype(np.float32)
        total_w = mask.sum()
        if total_w < 1:
            raw[i] = (src_w / 2.0, src_h / 2.0)
        else:
            ys, xs = np.mgrid[0:src_h, 0:src_w]
            raw[i] = (float((xs * mask).sum() / total_w),
                      float((ys * mask).sum() / total_w))
    cap.release()

    if not raw:
        return TrackResult(centres=[(src_w / 2.0, src_h / 2.0)] * total)

    # EMA over sampled keyframes
    ema_x = ema_y = None
    smoothed: dict[int, tuple[float, float]] = {}
    for i in sorted(raw):
        cx, cy = raw[i]
        if ema_x is None:
            ema_x, ema_y = cx, cy
        else:
            ema_x = ema_alpha * cx + (1 - ema_alpha) * ema_x
            ema_y = ema_alpha * cy + (1 - ema_alpha) * ema_y
        smoothed[i] = (ema_x, ema_y)

    keys = sorted(smoothed)
    centres: list[tuple[float, float]] = []
    for i in range(total):
        lo = max((k for k in keys if k <= i), default=keys[0])
        hi = min((k for k in keys if k >= i), default=keys[-1])
        if lo == hi:
            centres.append(smoothed[lo])
        else:
            t  = (i - lo) / (hi - lo)
            cx = smoothed[lo][0] + t * (smoothed[hi][0] - smoothed[lo][0])
            cy = smoothed[lo][1] + t * (smoothed[hi][1] - smoothed[lo][1])
            centres.append((cx, cy))

    return TrackResult(centres=centres)
