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


# ── Piece 3: Kalman Filter + Single-Face Reframing ────────────────────────────
#
# Pipeline for single_face clips:
#   1. Detect the largest face at 5 FPS → raw (cx, cy) centres
#   2. Fill gaps (frames with no detection) by holding the last known position
#   3. Smooth x and y independently with a 1-D Kalman filter
#   4. Interpolate the smoothed 5-FPS centres to every frame
#   5. Pipe frames through FFmpeg: crop 9:16 window centred on the face,
#      scale to 1080×1920, encode H.264 (NVENC if available)
#
# The Kalman filter is deliberately low-trust of new measurements
# (high measurement noise) so the crop window glides rather than jerks.

import subprocess


# ── GPU detection ─────────────────────────────────────────────────────────────

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


# ── 1-D Kalman filter ─────────────────────────────────────────────────────────

class _Kalman1D:
    """
    Minimal 1-D Kalman filter.
    State  = [position, velocity]
    Measurement = position only.

    process_var   — how much the true position can drift between frames (low = smooth)
    measure_var   — how much we trust each face-detection reading (high = smooth)
    """

    def __init__(self, process_var: float = 5.0, measure_var: float = 200.0):
        self._x  = None          # estimated position
        self._v  = 0.0           # estimated velocity
        self._p  = 1e4           # error covariance
        self._q  = process_var
        self._r  = measure_var

    def update(self, measurement: float) -> float:
        if self._x is None:
            self._x = measurement
            return measurement

        # Predict
        x_pred = self._x + self._v
        p_pred = self._p + self._q

        # Update
        k       = p_pred / (p_pred + self._r)
        self._x = x_pred + k * (measurement - x_pred)
        self._v = self._v + 0.3 * (self._x - x_pred)   # light velocity update
        self._p = (1 - k) * p_pred
        return self._x


# ── Face centre detection at detection_fps ────────────────────────────────────

def _face_centres_single(
    video_path: str,
    start: float,
    end: float,
    detection_fps: float = 5.0,
) -> list[tuple[float, float]]:
    """
    Return (cx, cy) pixel coordinates of the *largest* face detected,
    sampled at detection_fps, smoothed with Kalman, then interpolated
    to every frame.

    Returns a list of (cx, cy) with length == total_frames in [start, end].
    """
    cap      = cv2.VideoCapture(video_path)
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    total_frames = int((end - start) * src_fps)
    step_frames  = max(1, int(src_fps / detection_fps))

    # ── Pass 1: detect at detection_fps ──────────────────────────────────────
    raw_cx: dict[int, float] = {}   # frame_idx → pixel cx
    raw_cy: dict[int, float] = {}

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    with _mp_face.FaceDetection(min_detection_confidence=0.5) as detector:
        for local_idx in range(total_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if local_idx % step_frames != 0:
                continue

            faces = _detect_faces(frame, detector)
            if not faces:
                continue

            # Pick largest face (highest bounding-box area)
            best = max(
                faces,
                key=lambda d: (
                    d.location_data.relative_bounding_box.width
                    * d.location_data.relative_bounding_box.height
                ),
            )
            bb   = best.location_data.relative_bounding_box
            raw_cx[local_idx] = (bb.xmin + bb.width  / 2) * src_w
            raw_cy[local_idx] = (bb.ymin + bb.height / 2) * src_h

    cap.release()

    if not raw_cx:
        # No face found at all — return frame centres
        log.warning("_face_centres_single: no faces detected, using frame centre")
        return [(src_w / 2, src_h / 2)] * total_frames

    # ── Pass 2: fill gaps by propagating the last known detection ─────────────
    filled_cx = {}
    filled_cy = {}
    last_x, last_y = src_w / 2, src_h / 2
    for i in range(total_frames):
        if i in raw_cx:
            last_x, last_y = raw_cx[i], raw_cy[i]
        filled_cx[i] = last_x
        filled_cy[i] = last_y

    # ── Pass 3: Kalman smooth the detection keyframes ─────────────────────────
    kx, ky = _Kalman1D(), _Kalman1D()
    smoothed: dict[int, tuple[float, float]] = {}
    for i in sorted(filled_cx):
        smoothed[i] = (kx.update(filled_cx[i]), ky.update(filled_cy[i]))

    # ── Pass 4: linear interpolate to every frame ─────────────────────────────
    keyframes = sorted(smoothed)
    result    = []
    for i in range(total_frames):
        # find surrounding keyframes
        lo = max((k for k in keyframes if k <= i), default=keyframes[0])
        hi = min((k for k in keyframes if k >= i), default=keyframes[-1])
        if lo == hi:
            result.append(smoothed[lo])
        else:
            t = (i - lo) / (hi - lo)
            cx = smoothed[lo][0] + t * (smoothed[hi][0] - smoothed[lo][0])
            cy = smoothed[lo][1] + t * (smoothed[hi][1] - smoothed[lo][1])
            result.append((cx, cy))

    return result


# ── Frame rendering helper ────────────────────────────────────────────────────

def _crop_frame(
    frame: np.ndarray,
    cx: float,
    cy: float,
    crop_w: int,
    crop_h: int,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Crop a (crop_w × crop_h) window centred on (cx, cy), resize to (out_w × out_h)."""
    src_h, src_w = frame.shape[:2]

    x1 = int(cx - crop_w / 2)
    y1 = int(cy - crop_h / 2)
    # Clamp so the crop stays inside the frame
    x1 = max(0, min(x1, src_w - crop_w))
    y1 = max(0, min(y1, src_h - crop_h))

    cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def _open_ffmpeg_pipe(output_path: str, out_w: int, out_h: int, fps: float) -> subprocess.Popen:
    """Start an FFmpeg process that reads raw BGR frames from stdin and encodes to output_path."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", _ENCODER,
    ]
    if _ENCODER == "h264_nvenc":
        cmd += ["-preset", "p4", "-rc", "vbr", "-cq", "26"]
    else:
        cmd += ["-preset", "fast", "-crf", "23"]
    cmd += ["-pix_fmt", "yuv420p", output_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _mux_audio(video_only: str, source_video: str, start: float, duration: float, final: str):
    """Extract audio from source_video[start:start+duration] and mux into video_only → final."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start), "-t", str(duration), "-i", source_video,
            "-i", video_only,
            "-map", "1:v", "-map", "0:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            "-shortest", final,
        ],
        check=True,
        capture_output=True,
    )


def _add_watermark(frame: np.ndarray) -> np.ndarray:
    """Burn a subtle 'ViralClips' watermark into the bottom-centre of the frame."""
    h, w = frame.shape[:2]
    text  = "ViralClips"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = w / 800          # scale relative to frame width
    thick = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = (w - tw) // 2
    y = h - 28
    # Dark shadow then white text
    cv2.putText(frame, text, (x + 2, y + 2), font, scale, (0, 0, 0),     thick + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x,     y    ), font, scale, (220, 220, 220), thick,     cv2.LINE_AA)
    return frame


# ── Single-face reframe ───────────────────────────────────────────────────────

def reframe_single_face(
    video_path: str,
    start: float,
    end: float,
    output_path: str,
    watermark: bool = False,
) -> None:
    """
    Reframe clip [start, end] to 1080×1920 centred on the tracked face.
    Writes the final muxed file to output_path.
    """
    OUT_W, OUT_H = 1080, 1920

    cap     = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    # 9:16 crop dimensions — use full source height, derive width
    crop_h = src_h
    crop_w = int(src_h * OUT_W / OUT_H)
    if crop_w > src_w:          # source is already narrower than 9:16
        crop_w = src_w
        crop_h = int(src_w * OUT_H / OUT_W)

    duration     = end - start
    total_frames = int(duration * src_fps)

    log.info("single_face reframe: %.1fs clip, crop=%dx%d, encoder=%s", duration, crop_w, crop_h, _ENCODER)

    centres = _face_centres_single(video_path, start, end)

    # Video-only temp file (audio added in mux step)
    vid_tmp = str(WORK_DIR / f"{os.path.basename(output_path)}.vidonly.mp4")

    proc = _open_ffmpeg_pipe(vid_tmp, OUT_W, OUT_H, src_fps)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    for i in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        cx, cy = centres[i] if i < len(centres) else (src_w / 2, src_h / 2)
        out    = _crop_frame(frame, cx, cy, crop_w, crop_h, OUT_W, OUT_H)
        if watermark:
            out = _add_watermark(out)
        proc.stdin.write(out.tobytes())

    proc.stdin.close()
    proc.wait()
    cap.release()

    _mux_audio(vid_tmp, video_path, start, duration, output_path)
    os.unlink(vid_tmp)
    log.info("single_face done → %s", output_path)
