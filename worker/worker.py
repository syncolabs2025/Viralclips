"""
ViralClips Worker — pipeline orchestrator.

Reframing is handled by the reframe package (ReframeEngine).
When running inside a Modal container the engine instance is injected via
_REFRAME_ENGINE by modal_worker.py so models are loaded once per container.
Outside Modal the engine is constructed lazily on first use with no GPU models
(Kalman fallback only), which is useful for local development.
"""

import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import psycopg2
import psycopg2.extras
from redis import Redis
from rq import Queue

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("worker")

# ── Config ────────────────────────────────────────────────────────────────────

DATABASE_URL  = os.environ["DATABASE_URL"]
REDIS_URL     = os.environ["REDIS_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
FROM_EMAIL     = os.getenv("FROM_EMAIL", "noreply@viralclips.app")

GCS_BUCKET    = os.getenv("GCS_BUCKET", "")
LOCAL_UPLOAD_DIR = Path(os.getenv("LOCAL_UPLOAD_DIR", "/tmp/viralclips/uploads"))
LOCAL_OUTPUT_DIR = Path(os.getenv("LOCAL_OUTPUT_DIR", "/tmp/viralclips/outputs"))

# Work dir for this worker instance (temp files live here)
WORK_DIR = Path(tempfile.mkdtemp(prefix="vc_worker_"))

# ── Database ──────────────────────────────────────────────────────────────────

def _db_conn():
    """Open a fresh psycopg2 connection (each task opens and closes its own)."""
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def db_get_job(job_id: str) -> dict:
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT j.*, u.tier, u.email AS user_email
                FROM jobs j
                JOIN users u ON u.id = j.user_id
                WHERE j.id = %s
                """,
                (job_id,),
            )
            row = cur.fetchone()
    if not row:
        raise ValueError(f"Job {job_id} not found")
    return dict(row)


def db_update_job(job_id: str, **kwargs):
    """Update arbitrary job columns. Always bumps updated_at."""
    kwargs["updated_at"] = datetime.utcnow()
    cols = ", ".join(f"{k} = %s" for k in kwargs)
    vals = list(kwargs.values()) + [job_id]
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE jobs SET {cols} WHERE id = %s", vals)
        conn.commit()


def db_create_clip(job_id: str, start: float, end: float, reason: str, hook_score: int) -> str:
    clip_id = str(uuid.uuid4())
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO clips (id, job_id, start_time, end_time, reason, hook_score, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'pending')
                """,
                (clip_id, job_id, start, end, reason, hook_score),
            )
        conn.commit()
    return clip_id


def db_update_clip(clip_id: str, **kwargs):
    cols = ", ".join(f"{k} = %s" for k in kwargs)
    vals = list(kwargs.values()) + [clip_id]
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE clips SET {cols} WHERE id = %s", vals)
        conn.commit()


def db_increment_usage(user_id: str, billing_period: str, minutes: float):
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO usage_records (id, user_id, billing_period, videos_processed, minutes_processed)
                VALUES (%s, %s, %s, 1, %s)
                ON CONFLICT (user_id, billing_period)
                DO UPDATE SET
                    videos_processed  = usage_records.videos_processed  + 1,
                    minutes_processed = usage_records.minutes_processed + EXCLUDED.minutes_processed
                """,
                (str(uuid.uuid4()), user_id, billing_period, minutes),
            )
        conn.commit()

# ── GCS / Local storage ───────────────────────────────────────────────────────

def storage_download(storage_path: str, dest: Path) -> Path:
    """
    Download from GCS or local filesystem to dest.
    storage_path is either  gs://bucket/key  or  local://uploads/job/file.mp4
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if storage_path.startswith("local://"):
        rel  = storage_path.removeprefix("local://")
        src  = LOCAL_UPLOAD_DIR.parent / rel      # /tmp/viralclips/uploads/...
        import shutil
        shutil.copy2(str(src), str(dest))
        log.info("Copied local file %s → %s", src, dest)
        return dest

    # GCS
    from google.cloud import storage as gcs
    client    = gcs.Client()
    blob_name = storage_path.removeprefix(f"gs://{GCS_BUCKET}/")
    blob      = client.bucket(GCS_BUCKET).blob(blob_name)
    blob.download_to_filename(str(dest))
    log.info("Downloaded GCS %s → %s (%.1f MB)", blob_name, dest, dest.stat().st_size/1e6)
    return dest


def storage_upload(local_path: Path, storage_path: str) -> str:
    """
    Upload local_path to GCS or local filesystem.
    Returns the storage_path it was stored at.
    """
    if not GCS_BUCKET:
        rel  = storage_path.removeprefix("local://")
        dest = LOCAL_OUTPUT_DIR.parent / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(str(local_path), str(dest))
        log.info("Stored locally: %s", dest)
        return storage_path

    from google.cloud import storage as gcs
    client    = gcs.Client()
    blob_name = storage_path.removeprefix(f"gs://{GCS_BUCKET}/")
    blob      = client.bucket(GCS_BUCKET).blob(blob_name)
    blob.upload_from_filename(str(local_path))
    log.info("Uploaded to GCS: %s (%.1f MB)", blob_name, local_path.stat().st_size/1e6)
    return f"gs://{GCS_BUCKET}/{blob_name}"


# ── Reframe engine ────────────────────────────────────────────────────────────
#
# _REFRAME_ENGINE is injected by modal_worker.py when running inside a Modal
# container (models already loaded).  Locally it is constructed on first use
# with no GPU models so development still works without a GPU.

import cv2
import numpy as np

from reframe import ContentCategory, ReframeEngine

# Injected by modal_worker.py; constructed lazily for local dev (CPU/Kalman only)
_REFRAME_ENGINE: ReframeEngine | None = None


def _get_engine() -> ReframeEngine:
    global _REFRAME_ENGINE
    if _REFRAME_ENGINE is None:
        _REFRAME_ENGINE = ReframeEngine()   # no models — Kalman fallback
    return _REFRAME_ENGINE


# ── Piece 6: Main Pipeline ────────────────────────────────────────────────────
#
# process_job(job_id) is the single function enqueued by the API server.
# It runs the full pipeline end-to-end and updates Postgres at every step.
#
# Error contract:
#   - Top-level exceptions mark the whole job failed.
#   - Per-clip exceptions are caught; that clip is marked failed and skipped.
#     The job continues with remaining clips and still produces a zip.
#   - If zero clips succeed the job is marked failed.

import json
import shutil
import zipfile
from openai import OpenAI


_openai = OpenAI(api_key=OPENAI_API_KEY)

VIRAL_PROMPT = """You are a viral short-form content strategist.
Given a video transcript with timestamps:

1. Classify the overall content type as exactly one of:
   solo_vlog, podcast, interview, music_video, gaming, travel_scenery,
   comedy_skit, fitness, sports_action, cooking_food, product_review,
   keynote_talk, reaction_video, asmr_lofi, unknown

2. Identify the 5 most viral-worthy moments. Each must be 10–30 seconds long
   and work as a standalone clip.
   Focus on: strong hooks, emotional peaks, surprising statements,
   cliffhangers, quotable moments.

Return a single JSON object:
{
  "content_category": "<one of the categories above>",
  "clips": [
    {
      "start_time": <float seconds>,
      "end_time":   <float seconds>,
      "reason":     "<one sentence why this is viral>",
      "hook_score": <integer 1-10>
    }
  ]
}

Respond with valid JSON only — no markdown, no explanation."""


# ── Step 1: Download ──────────────────────────────────────────────────────────

def _download_video(job: dict) -> Path:
    ext       = Path(job["original_filename"]).suffix or ".mp4"
    dest      = WORK_DIR / f"source{ext}"
    db_update_job(job["id"], status="downloading")
    storage_download(job["gcs_input_path"], dest)
    log.info("Downloaded video: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


# ── Step 2: Transcribe ────────────────────────────────────────────────────────

def _transcribe(job: dict, video_path: Path) -> dict:
    db_update_job(job["id"], status="transcribing")
    log.info("Transcribing via Whisper API…")

    with open(video_path, "rb") as f:
        response = _openai.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    transcript = {
        "text":     response.text,
        "segments": [
            {
                "id":    s.id,
                "start": s.start,
                "end":   s.end,
                "text":  s.text.strip(),
            }
            for s in (response.segments or [])
        ],
    }

    # Persist transcript + video duration to DB
    duration_min = (transcript["segments"][-1]["end"] / 60) if transcript["segments"] else 0
    db_update_job(job["id"], transcript=json.dumps(transcript))

    # Update usage now that we know the duration
    period = datetime.utcnow().strftime("%Y-%m")
    db_increment_usage(job["user_id"], period, duration_min)

    log.info(
        "Transcribed: %d segments, %.1f min",
        len(transcript["segments"]), duration_min,
    )
    return transcript


# ── Step 3: Detect viral moments ──────────────────────────────────────────────

def _detect_moments(job: dict, transcript: dict) -> tuple[list[dict], ContentCategory]:
    """
    Returns (clips, content_category).
    GPT-4o mini now classifies the video type alongside detecting moments —
    zero extra API cost since it reads the same transcript either way.
    """
    db_update_job(job["id"], status="detecting")
    log.info("Detecting viral moments + content category with GPT-4o mini…")

    formatted = "\n".join(
        f"[{s['start']:.1f}s – {s['end']:.1f}s] {s['text']}"
        for s in transcript["segments"]
    )

    response = _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": VIRAL_PROMPT},
            {"role": "user",   "content": formatted},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    parsed   = json.loads(response.choices[0].message.content)
    clips    = parsed.get("clips", [])
    cat_raw  = parsed.get("content_category", "unknown")

    try:
        category = ContentCategory(cat_raw)
    except ValueError:
        log.warning("Unknown content_category %r — defaulting to unknown", cat_raw)
        category = ContentCategory.UNKNOWN

    # Clamp to transcript bounds and drop clips shorter than 8 s
    max_t = transcript["segments"][-1]["end"] if transcript["segments"] else 9999
    valid = []
    for c in clips:
        s, e = float(c["start_time"]), float(c["end_time"])
        e    = min(e, max_t)
        if e - s >= 8:
            valid.append({**c, "start_time": s, "end_time": e})

    log.info("Detected %d valid clips — category: %s", len(valid), category.value)
    return valid, category


# ── Step 4: Reframe one clip ──────────────────────────────────────────────────

def _reframe_clip(
    job:         dict,
    clip_id:     str,
    video_path:  Path,
    start:       float,
    end:         float,
    category:    ContentCategory,
    output_path: Path,
) -> None:
    """Delegate to ReframeEngine. Raises on failure; caller skips the clip."""
    _get_engine().process(
        video_path=str(video_path),
        start=start,
        end=end,
        category=category,
        output_path=str(output_path),
        watermark=(job["tier"] == "free"),
    )


# ── Step 5: Zip + upload ──────────────────────────────────────────────────────

def _zip_and_upload(job: dict, clip_paths: list[tuple[str, Path]]) -> str:
    """
    Zip all successful clip files, upload to GCS (or local), return storage path.
    clip_paths: list of (arcname, local_path) tuples.
    """
    db_update_job(job["id"], status="zipping")

    zip_local = WORK_DIR / f"{job['id']}_clips.zip"
    with zipfile.ZipFile(zip_local, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in clip_paths:
            zf.write(path, arcname)
    log.info("Zipped %d clips → %.1f MB", len(clip_paths), zip_local.stat().st_size / 1e6)

    storage_path = (
        f"gs://{GCS_BUCKET}/outputs/{job['id']}/clips.zip"
        if GCS_BUCKET
        else f"local://outputs/{job['id']}/clips.zip"
    )
    return storage_upload(zip_local, storage_path)


# ── Step 6: Email notification ────────────────────────────────────────────────

def _notify(job: dict, n_clips: int):
    if not RESEND_API_KEY or not job.get("user_email"):
        return
    try:
        import resend  # noqa: PLC0415

        resend.api_key = RESEND_API_KEY
        resend.Emails.send({
            "from":    FROM_EMAIL,
            "to":      [job["user_email"]],
            "subject": f"Your ViralClips are ready ({n_clips} clips)",
            "html":    (
                f"<p>Hi,</p>"
                f"<p>Your video <strong>{job['original_filename']}</strong> has been processed.</p>"
                f"<p>{n_clips} clip(s) reframed to 9:16 are ready to download.</p>"
                f"<p><a href='https://viralclips.app/jobs/{job['id']}'>View &amp; Download</a></p>"
            ),
        })
        log.info("Email sent to %s", job["user_email"])
    except Exception as exc:
        log.warning("Email notification failed: %s", exc)   # non-fatal


# ── Master orchestrator ───────────────────────────────────────────────────────

def process_job(job_id: str):
    """
    Full pipeline. Called by RQ worker.
    Stateless — safe to kill and retry (job status resets on restart via max_retries).
    """
    log.info("▶ Starting job %s", job_id)
    job = db_get_job(job_id)

    try:
        # ── 1. Download ───────────────────────────────────────────────────────
        video_path = _download_video(job)

        # ── 2. Transcribe ─────────────────────────────────────────────────────
        transcript = _transcribe(job, video_path)
        if not transcript["segments"]:
            raise ValueError("Whisper returned an empty transcript")

        # ── 3. Detect viral moments + classify content category ───────────────
        moments, category = _detect_moments(job, transcript)
        if not moments:
            raise ValueError("GPT-4o mini returned no valid viral moments")

        log.info("Content category: %s", category.value)

        # Create clip rows and update total count
        clip_ids = []
        for m in moments:
            cid = db_create_clip(
                job_id     = job_id,
                start      = m["start_time"],
                end        = m["end_time"],
                reason     = m["reason"],
                hook_score = int(m.get("hook_score", 5)),
            )
            clip_ids.append((cid, m))

        db_update_job(job_id, status="reframing", clips_total=len(clip_ids))

        # ── 4. Reframe each clip (skip individual failures) ───────────────────
        successful_clips: list[tuple[str, Path]] = []

        for idx, (clip_id, moment) in enumerate(clip_ids, start=1):
            start, end = moment["start_time"], moment["end_time"]
            log.info("Clip %d/%d  [%.1fs–%.1fs]  category=%s",
                     idx, len(clip_ids), start, end, category.value)

            try:
                db_update_clip(clip_id, status="processing", content_type=category.value)

                out_path = WORK_DIR / f"clip_{idx:02d}_{category.value}.mp4"
                _reframe_clip(job, clip_id, video_path, start, end, category, out_path)

                db_update_clip(clip_id, status="complete", output_path=str(out_path))
                db_update_job(job_id, clips_done=idx)

                arcname = (
                    f"clip_{idx:02d}_{int(start)}s-{int(end)}s"
                    f"_{category.value}"
                    f"_score{moment.get('hook_score','?')}.mp4"
                )
                successful_clips.append((arcname, out_path))

            except Exception as exc:
                log.error("Clip %d failed — skipping: %s", idx, exc, exc_info=True)
                db_update_clip(clip_id, status="failed")
                # Continue with remaining clips

        if not successful_clips:
            raise RuntimeError("Every clip failed during reframing")

        # ── 5. Zip + upload ───────────────────────────────────────────────────
        output_storage = _zip_and_upload(job, successful_clips)
        db_update_job(
            job_id,
            status          = "complete",
            gcs_output_path = output_storage,
            clips_done      = len(successful_clips),
        )

        # ── 6. Notify ─────────────────────────────────────────────────────────
        _notify(job, len(successful_clips))
        log.info("✓ Job %s complete — %d clips", job_id, len(successful_clips))

    except Exception as exc:
        log.error("✗ Job %s failed: %s", job_id, exc, exc_info=True)
        db_update_job(job_id, status="failed", error=str(exc))
        raise   # re-raise so RQ marks the job as failed and respects max_retries

    finally:
        # Clean up per-job work directory (video + intermediate files)
        shutil.rmtree(str(WORK_DIR), ignore_errors=True)


# ── RQ Worker entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from rq import Worker

    log.info("Worker starting — queue=viralclips")

    redis_conn = Redis.from_url(REDIS_URL)
    queues     = sys.argv[1:] or ["viralclips"]

    w = Worker(queues, connection=redis_conn)
    w.work(with_scheduler=False)
