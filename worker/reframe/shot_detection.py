"""
Shot boundary detection: TransNetV2 (neural) + PySceneDetect (adaptive) union.
Both detectors run independently; their cut-point outputs are merged and
deduplicated within a 0.5 s window so a single cut is never counted twice.
"""

import logging
from dataclasses import dataclass

import cv2

log = logging.getLogger("reframe.shots")


@dataclass
class Shot:
    index:       int
    start_frame: int
    end_frame:   int
    start_time:  float
    end_time:    float
    fps:         float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


# ── Per-detector helpers ──────────────────────────────────────────────────────

def _pyscenedetect(video_path: str, start: float, end: float) -> list[float]:
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import AdaptiveDetector

        video    = open_video(video_path)
        mgr      = SceneManager()
        mgr.add_detector(AdaptiveDetector(adaptive_threshold=3.0))
        mgr.detect_scenes(video, show_progress=False)

        return [
            s[0].get_seconds()
            for s in mgr.get_scene_list()
            if start < s[0].get_seconds() < end
        ]
    except Exception as exc:
        log.warning("PySceneDetect failed: %s", exc)
        return []


def _transnetv2(video_path: str, start: float, end: float, fps: float) -> list[float]:
    try:
        from transnetv2 import TransNetV2

        model = TransNetV2()
        _, single_pred, _ = model.predict_video(video_path)
        scenes = model.predictions_to_scenes(single_pred, threshold=0.5)
        return [
            s[0] / fps
            for s in scenes
            if start < s[0] / fps < end
        ]
    except ImportError:
        log.debug("TransNetV2 not installed — using PySceneDetect only")
        return []
    except Exception as exc:
        log.warning("TransNetV2 failed: %s", exc)
        return []


# ── Public API ────────────────────────────────────────────────────────────────

def detect_shots(video_path: str, start: float, end: float) -> list[Shot]:
    """
    Return shots within [start, end] using a union of TransNetV2 and
    PySceneDetect.  Cuts within 0.5 s of each other are merged.
    Sub-200 ms shots (likely detection noise) are dropped.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    cuts_a = _transnetv2(video_path, start, end, fps)
    cuts_b = _pyscenedetect(video_path, start, end)

    # Merge and deduplicate within 0.5 s window
    all_cuts: list[float] = sorted(set(cuts_a + cuts_b))
    merged: list[float] = []
    for t in all_cuts:
        if not merged or (t - merged[-1]) > 0.5:
            merged.append(t)

    boundaries = [start] + merged + [end]
    shots: list[Shot] = []
    for i in range(len(boundaries) - 1):
        t0, t1 = boundaries[i], boundaries[i + 1]
        if t1 - t0 < 0.2:
            continue
        shots.append(Shot(
            index=len(shots),
            start_frame=int(t0 * fps),
            end_frame=int(t1 * fps),
            start_time=t0,
            end_time=t1,
            fps=fps,
        ))

    if not shots:
        shots = [Shot(0, int(start * fps), int(end * fps), start, end, fps)]

    log.info(
        "Shot detection: %d shots in [%.1fs–%.1fs] (cuts: transnetv2=%d psd=%d merged=%d)",
        len(shots), start, end, len(cuts_a), len(cuts_b), len(merged),
    )
    return shots
