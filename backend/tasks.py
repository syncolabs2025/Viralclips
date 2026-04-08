"""
Celery tasks for the ViralClips pipeline.

Task chain per job:
  transcribe_video → detect_viral_moments → reframe_clip (one per clip, parallel)
                                          → check_and_finalize (after each clip)
"""

import base64
import json
import logging
import os
import subprocess
import time
import zipfile
from datetime import datetime
from pathlib import Path

import requests
import whisper
from openai import OpenAI

from celery_app import celery
from database import SessionLocal
from models import Clip, Job

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
CLIPS_DIR = Path(os.getenv("CLIPS_DIR", "./clips"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

RUNWAY_BASE_URL = "https://api.runwayml.com/v1"
# Runway Gen-3 supports 5 s or 10 s output per task
RUNWAY_MAX_SECONDS = 10

_openai = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db():
    return SessionLocal()


def _fail_job(db, job_id: str, error: str):
    db.query(Job).filter(Job.id == job_id).update(
        {"status": "failed", "error": error, "updated_at": datetime.utcnow()}
    )
    db.commit()


def _runway_headers() -> dict:
    return {
        "Authorization": f"Bearer {RUNWAY_API_KEY}",
        "X-Runway-Version": "2024-11-06",
        "Content-Type": "application/json",
    }


def _video_to_data_uri(path: Path) -> str:
    """Base64-encode a local video file for inline Runway upload."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:video/mp4;base64,{data}"


def _video_source(path: Path, clip_id: str) -> str:
    """Return a URL or data-URI that Runway can fetch the clip from."""
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}/clips/{clip_id}.mp4"
    return _video_to_data_uri(path)


def _extract_clip(source_video: str, start: float, duration: float, dest: Path):
    """Use FFmpeg to cut a segment from the source video."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", source_video,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )


def _runway_reframe_segment(video_source: str, duration: float) -> str:
    """
    Submit one ≤10-second clip to Runway's video_to_video endpoint requesting
    9:16 portrait output. Returns the output video URL on success.

    Runway API reference: https://docs.runwayml.com/docs/video-to-video
    The ratio "768:1280" corresponds to 9:16 at Runway's standard resolution.
    """
    payload = {
        "model": "gen3a_turbo",
        "promptVideo": video_source,
        "prompt": (
            "Reframe to 9:16 vertical portrait format. "
            "Expand the canvas to fill the frame, keep the subject centred, "
            "mirror or outpaint the background naturally. "
            "Social media Reels / Shorts style."
        ),
        "ratio": "768:1280",
        "duration": min(int(duration) or 5, RUNWAY_MAX_SECONDS),
    }

    resp = requests.post(
        f"{RUNWAY_BASE_URL}/video_to_video",
        headers=_runway_headers(),
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    task_id = resp.json()["id"]

    # Poll until terminal state (max 12 minutes)
    for _ in range(144):
        time.sleep(5)
        poll = requests.get(
            f"{RUNWAY_BASE_URL}/tasks/{task_id}",
            headers=_runway_headers(),
            timeout=30,
        )
        poll.raise_for_status()
        data = poll.json()
        status = data.get("status", "")

        if status == "SUCCEEDED":
            return data["output"][0]
        if status in ("FAILED", "CANCELLED"):
            raise RuntimeError(
                f"Runway task {task_id} {status}: {data.get('failure', 'unknown')}"
            )

    raise TimeoutError(f"Runway task {task_id} timed out after 12 minutes")


def _download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)


# ---------------------------------------------------------------------------
# Task 1 — Transcribe
# ---------------------------------------------------------------------------


@celery.task(bind=True, max_retries=3, default_retry_delay=60)
def transcribe_video(self, job_id: str):
    db = _db()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return

        job.status = "transcribing"
        job.updated_at = datetime.utcnow()
        db.commit()

        log.info("Loading Whisper model for job %s", job_id)
        model = whisper.load_model("base")
        result = model.transcribe(job.video_path, verbose=False)

        transcript_data = {
            "text": result["text"],
            "segments": [
                {
                    "id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                for seg in result["segments"]
            ],
        }

        job.transcript = json.dumps(transcript_data)
        job.status = "detecting"
        job.updated_at = datetime.utcnow()
        db.commit()

        detect_viral_moments.delay(job_id)

    except Exception as exc:
        log.exception("transcribe_video failed for job %s", job_id)
        _fail_job(db, job_id, str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Task 2 — Detect viral moments
# ---------------------------------------------------------------------------


@celery.task(bind=True, max_retries=3, default_retry_delay=30)
def detect_viral_moments(self, job_id: str):
    db = _db()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return

        transcript_data = json.loads(job.transcript)
        total_duration = (
            transcript_data["segments"][-1]["end"]
            if transcript_data["segments"]
            else 0
        )

        formatted = "\n".join(
            f"[{s['start']:.1f}s – {s['end']:.1f}s] {s['text']}"
            for s in transcript_data["segments"]
        )

        prompt = f"""You are a viral short-form content strategist.
Analyse this video transcript (total duration: {total_duration:.0f}s) and identify
3–7 moments that would perform best as standalone Reels or YouTube Shorts.

Each clip must:
- Be 15–60 seconds long
- Have a strong hook, punchline, surprising insight, or emotional moment
- Stand alone without needing context from the rest of the video

Transcript (with timestamps):
{formatted}

Reply with valid JSON only — no markdown, no explanation:
{{
  "clips": [
    {{
      "start_time": <float seconds>,
      "end_time":   <float seconds>,
      "reason":     "<one sentence: why this moment is viral-worthy>"
    }}
  ]
}}"""

        response = _openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        result = json.loads(response.choices[0].message.content)
        clips_data = result.get("clips") or []

        if not clips_data:
            _fail_job(db, job_id, "GPT-4o mini returned no viral moments")
            return

        clip_ids = []
        for c in clips_data:
            start = float(c["start_time"])
            end = float(c["end_time"])
            # Clamp to transcript bounds
            end = min(end, total_duration or end)
            if end - start < 5:
                continue
            clip = Clip(
                job_id=job_id,
                start_time=start,
                end_time=end,
                reason=c["reason"],
                status="pending",
            )
            db.add(clip)
            db.flush()
            clip_ids.append(clip.id)

        if not clip_ids:
            _fail_job(db, job_id, "All detected clips were too short (<5 s)")
            return

        job.status = "reframing"
        job.updated_at = datetime.utcnow()
        db.commit()

        for clip_id in clip_ids:
            reframe_clip.delay(job_id, clip_id)

    except Exception as exc:
        log.exception("detect_viral_moments failed for job %s", job_id)
        _fail_job(db, job_id, str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Task 3 — Reframe one clip with Runway ML
# ---------------------------------------------------------------------------


@celery.task(bind=True, max_retries=3, default_retry_delay=90)
def reframe_clip(self, job_id: str, clip_id: str):
    db = _db()
    try:
        clip = db.query(Clip).filter(Clip.id == clip_id).first()
        job = db.query(Job).filter(Job.id == job_id).first()
        if not clip or not job:
            return

        clip.status = "processing"
        db.commit()

        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        duration = clip.end_time - clip.start_time

        # ── Step 1: Extract segment with FFmpeg ──────────────────────────────
        clip_path = CLIPS_DIR / f"{clip_id}.mp4"
        _extract_clip(job.video_path, clip.start_time, duration, clip_path)

        # ── Step 2: Handle clips longer than Runway's 10-second limit ────────
        # Split into ≤10 s chunks, reframe each, then concatenate.
        segment_paths = []

        if duration <= RUNWAY_MAX_SECONDS:
            chunks = [(clip_path, duration)]
        else:
            chunks = []
            offset = 0.0
            idx = 0
            while offset < duration:
                seg_dur = min(RUNWAY_MAX_SECONDS, duration - offset)
                seg_path = CLIPS_DIR / f"{clip_id}_seg{idx}.mp4"
                _extract_clip(str(clip_path), offset, seg_dur, seg_path)
                chunks.append((seg_path, seg_dur))
                offset += seg_dur
                idx += 1

        # ── Step 3: Submit each chunk to Runway ML ───────────────────────────
        reframed_paths = []
        for seg_path, seg_dur in chunks:
            video_src = _video_source(seg_path, clip_id)
            output_url = _runway_reframe_segment(video_src, seg_dur)

            out_path = OUTPUT_DIR / f"{clip_id}_{seg_path.stem}_9x16.mp4"
            _download_file(output_url, out_path)
            reframed_paths.append(out_path)

        # ── Step 4: Concatenate segments if needed ────────────────────────────
        final_path = OUTPUT_DIR / f"{clip_id}_9x16.mp4"

        if len(reframed_paths) == 1:
            reframed_paths[0].rename(final_path)
        else:
            concat_list = OUTPUT_DIR / f"{clip_id}_concat.txt"
            concat_list.write_text(
                "\n".join(f"file '{p.resolve()}'" for p in reframed_paths)
            )
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_list),
                    "-c", "copy",
                    str(final_path),
                ],
                check=True,
                capture_output=True,
            )
            concat_list.unlink(missing_ok=True)

        clip.output_path = str(final_path)
        clip.status = "complete"
        db.commit()

        # Trigger finalization check (idempotent — only zips when all done)
        check_and_finalize.delay(job_id)

    except Exception as exc:
        log.exception("reframe_clip failed for clip %s", clip_id)
        db.query(Clip).filter(Clip.id == clip_id).update({"status": "failed"})
        db.commit()
        # Still check finalization — other clips may have succeeded
        check_and_finalize.delay(job_id)
        raise self.retry(exc=exc)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Task 4 — Finalize: zip outputs and mark job complete
# ---------------------------------------------------------------------------


@celery.task
def check_and_finalize(job_id: str):
    db = _db()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job or job.status not in ("reframing",):
            return

        clips = db.query(Clip).filter(Clip.job_id == job_id).all()
        statuses = {c.status for c in clips}

        # Still waiting for at least one clip to finish
        if "pending" in statuses or "processing" in statuses:
            return

        done_clips = [c for c in clips if c.status == "complete" and c.output_path]
        if not done_clips:
            _fail_job(db, job_id, "All clip reframing tasks failed")
            return

        zip_path = OUTPUT_DIR / f"{job_id}_clips.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, clip in enumerate(done_clips, start=1):
                arcname = (
                    f"clip_{i:02d}_{int(clip.start_time)}s-{int(clip.end_time)}s.mp4"
                )
                zf.write(clip.output_path, arcname)

        job.status = "complete"
        job.updated_at = datetime.utcnow()
        db.commit()
        log.info("Job %s complete — %d clips zipped", job_id, len(done_clips))

    finally:
        db.close()
