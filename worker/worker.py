"""
ViralClips Worker — Pieces 1 + 2: Config, DB, GCS, Content Classification
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
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
FROM_EMAIL    = os.getenv("FROM_EMAIL", "noreply@viralclips.app")

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


# ── Piece 2: Content Classification ──────────────────────────────────────────
#
# Strategy:
#   Sample 10 evenly-spaced frames from the clip segment.
#   Run MediaPipe Face Detection on each frame.
#   Average the face count across samples:
#     avg >= 1.8  →  dual_face   (podcast / interview — two people visible)
#     avg >= 0.5  →  single_face (vlog, tutorial, talking head)
#     avg <  0.5  →  no_face     (scenery, screen share, gameplay)
#
# We keep the MediaPipe detector alive for the whole classification pass
# (creating it once is much faster than per-frame init).

import cv2
import mediapipe as mp
import numpy as np

_mp_face = mp.solutions.face_detection


def _detect_faces(frame_bgr: np.ndarray, detector) -> list:
    """Run MediaPipe face detection on one BGR frame. Returns list of detections."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)
    return result.detections or []


def _sample_frames(video_path: str, start: float, end: float, n: int = 10) -> list:
    """
    Return n evenly-spaced BGR frames from [start, end] seconds of video_path.
    Skips frames that can't be decoded rather than crashing.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []

    for i in range(n):
        t = start + (end - start) * i / max(n - 1, 1)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)

    cap.release()
    return frames


def classify_content(video_path: str, start: float, end: float) -> str:
    """
    Return one of: 'single_face' | 'dual_face' | 'no_face'

    Samples 10 frames from the clip window and counts faces per frame
    using MediaPipe. The average count determines the layout strategy.
    """
    frames = _sample_frames(video_path, start, end, n=10)
    if not frames:
        log.warning("classify_content: no frames decoded — defaulting to no_face")
        return "no_face"

    face_counts = []
    with _mp_face.FaceDetection(min_detection_confidence=0.5) as detector:
        for frame in frames:
            faces = _detect_faces(frame, detector)
            face_counts.append(len(faces))

    avg = sum(face_counts) / len(face_counts)
    log.info(
        "classify_content [%.1fs–%.1fs]: face counts=%s avg=%.2f",
        start, end, face_counts, avg,
    )

    if avg >= 1.8:
        return "dual_face"
    if avg >= 0.5:
        return "single_face"
    return "no_face"
