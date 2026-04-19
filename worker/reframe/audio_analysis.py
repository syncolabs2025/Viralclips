"""
Audio analysis:
  - pyannote/speaker-diarization-3.1  → deterministic per-speaker timestamps
  - librosa beat_track                → beat grid for music-video cut snapping
Both are optional — the pipeline degrades gracefully when they are unavailable.
"""

import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

log = logging.getLogger("reframe.audio")


@dataclass
class AudioAnalysis:
    # {speaker_label: [(start_sec, end_sec), ...]}  — empty when diarization unavailable
    speaker_segments: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    # sorted beat timestamps in seconds — empty when librosa unavailable / not music
    beat_times: list[float] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_wav(video_path: str) -> str:
    """Extract 16 kHz mono WAV from video to a temp file. Returns path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vn", "-ar", "16000", "-ac", "1", tmp.name],
        check=True, capture_output=True,
    )
    return tmp.name


# ── Speaker diarization ───────────────────────────────────────────────────────

def run_diarization(video_path: str) -> dict[str, list[tuple[float, float]]]:
    """
    Pyannote speaker diarization.  Returns {speaker_id: [(start, end), ...]}.
    Falls back to {} when pyannote is unavailable or the HF token is missing.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        log.info("HF_TOKEN not set — diarization skipped")
        return {}

    try:
        import torch
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        wav = _to_wav(video_path)
        try:
            dia = pipeline(wav)
        finally:
            os.unlink(wav)

        speakers: dict[str, list[tuple[float, float]]] = {}
        for turn, _, label in dia.itertracks(yield_label=True):
            speakers.setdefault(label, []).append((turn.start, turn.end))

        log.info("Diarization complete: %d speakers", len(speakers))
        return speakers

    except ImportError:
        log.info("pyannote.audio not installed — diarization skipped")
        return {}
    except Exception as exc:
        log.warning("Diarization failed: %s", exc)
        return {}


# ── Beat detection ────────────────────────────────────────────────────────────

def run_beat_detection(video_path: str) -> list[float]:
    """
    Librosa beat tracking.  Returns sorted beat timestamps in seconds.
    Falls back to [] when librosa is unavailable.
    """
    try:
        import librosa

        wav = _to_wav(video_path)
        try:
            y, sr = librosa.load(wav, sr=22050, mono=True)
        finally:
            os.unlink(wav)

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        log.info("Beat detection: %.1f BPM, %d beats", float(tempo), len(beat_times))
        return beat_times

    except ImportError:
        log.info("librosa not installed — beat detection skipped")
        return []
    except Exception as exc:
        log.warning("Beat detection failed: %s", exc)
        return []


# ── Public entry point ────────────────────────────────────────────────────────

def analyze_audio(video_path: str, is_music: bool = False) -> AudioAnalysis:
    """Run diarization (always) and beat detection (only for music_video) in parallel."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_dia  = pool.submit(run_diarization, video_path)
        f_beat = pool.submit(run_beat_detection, video_path) if is_music else None

        speakers   = f_dia.result()
        beat_times = f_beat.result() if f_beat else []

    return AudioAnalysis(speaker_segments=speakers, beat_times=beat_times)


def active_speaker_at(t: float, segments: dict[str, list[tuple[float, float]]]) -> str | None:
    """Return the speaker label active at time t, or None if silence / unavailable."""
    for label, segs in segments.items():
        for s, e in segs:
            if s <= t <= e:
                return label
    return None


def nearest_beat(t: float, beat_times: list[float]) -> float:
    """Return the beat timestamp closest to t."""
    if not beat_times:
        return t
    return min(beat_times, key=lambda b: abs(b - t))
