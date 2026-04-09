"""
ViralClips — FastAPI backend

Endpoints:
  POST /auth/register
  POST /auth/login
  GET  /auth/me
  GET  /auth/verify-email/{token}
  POST /auth/resend-verification

  POST /upload/presign          → {job_id, upload_url, gcs_path}
  POST /jobs/{id}/confirm       → trigger worker after client finishes upload
  GET  /jobs                    → list user jobs
  GET  /jobs/{id}               → job status + clips
  GET  /jobs/{id}/download      → redirect to signed download URL
  GET  /billing                 → current usage

  # Local dev only
  PUT  /internal/upload/{id}    → receive raw video bytes (no GCS)
  GET  /internal/download/{path:path}
"""

import logging
import os
import secrets
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import redis as _redis
import rq
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from auth import create_token, get_current_user, hash_password, verify_password
from database import SessionLocal, engine, get_db
from models import Base, Clip, Job, UsageRecord, User
from schemas import (
    ClipOut, JobOut, LoginRequest, PresignRequest, PresignResponse,
    RegisterRequest, STATUS_LABELS, TokenResponse, UsageOut, UserOut,
)
from storage import (
    LOCAL_OUTPUT_DIR, LOCAL_UPLOAD_DIR, MAX_UPLOAD_BYTES,
    generate_download_url, generate_upload_url,
)

log = logging.getLogger("api")
Base.metadata.create_all(bind=engine)

LOCAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ViralClips", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Allowed file types ────────────────────────────────────────────────────────
# Both MIME type AND extension must match — prevents renaming .exe to .mp4

ALLOWED_MIME_TYPES = {
    "video/mp4", "video/quicktime", "video/x-msvideo",
    "video/x-matroska", "video/webm", "video/mpeg",
    "video/3gpp", "video/x-flv", "video/x-ms-wmv",
}

ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".mpeg", ".mpg", ".3gp", ".flv", ".wmv",
}

# ── Redis / RQ ────────────────────────────────────────────────────────────────

_redis_conn = _redis.from_url(os.environ["REDIS_URL"])
_queue      = rq.Queue("viralclips", connection=_redis_conn)


def enqueue_job(job_id: str):
    _queue.enqueue("worker.process_job", job_id, job_timeout=3600)


# ── Email helpers ─────────────────────────────────────────────────────────────

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
FROM_EMAIL       = os.getenv("FROM_EMAIL", "noreply@viralclips.app")
API_BASE_URL     = os.getenv("API_BASE_URL", "http://localhost:8000")


def _send_verification_email(email: str, token: str):
    verify_url = f"{API_BASE_URL}/auth/verify-email/{token}"

    if not SENDGRID_API_KEY:
        # Local dev: just log the link — no email sent
        log.info("EMAIL VERIFICATION (dev mode) → %s", verify_url)
        return

    try:
        import sendgrid                          # noqa: PLC0415
        from sendgrid.helpers.mail import Mail  # noqa: PLC0415

        sg  = sendgrid.SendGridAPIClient(SENDGRID_API_KEY)
        msg = Mail(
            from_email   = FROM_EMAIL,
            to_emails    = email,
            subject      = "Verify your ViralClips account",
            html_content = (
                f"<p>Welcome to ViralClips!</p>"
                f"<p><a href='{verify_url}'>Click here to verify your email</a></p>"
                f"<p>This link does not expire.</p>"
            ),
        )
        sg.send(msg)
    except Exception as exc:
        log.warning("Failed to send verification email to %s: %s", email, exc)


# ── Quota / billing helpers ───────────────────────────────────────────────────

FREE_VIDEO_LIMIT  = 3
PAID_MINUTE_LIMIT = 60.0


def _current_period() -> str:
    return datetime.utcnow().strftime("%Y-%m")


def _get_usage(db: Session, user_id: str) -> UsageRecord:
    period = _current_period()
    usage  = (
        db.query(UsageRecord)
        .filter(UsageRecord.user_id == user_id, UsageRecord.billing_period == period)
        .first()
    )
    if not usage:
        usage = UsageRecord(user_id=user_id, billing_period=period)
        db.add(usage)
        db.commit()
        db.refresh(usage)
    return usage


def _check_quota(user: User, db: Session):
    usage = _get_usage(db, user.id)
    if user.tier == "free" and usage.videos_processed >= FREE_VIDEO_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Free tier limit reached ({FREE_VIDEO_LIMIT} videos/period). Upgrade to paid.",
        )


def _job_to_out(job: Job, clips: list) -> JobOut:
    clip_outs = [
        ClipOut(
            id=c.id,
            start_time=c.start_time,
            end_time=c.end_time,
            duration=round(c.end_time - c.start_time, 1),
            reason=c.reason,
            hook_score=c.hook_score,
            content_type=c.content_type,
            status=c.status,
        )
        for c in sorted(clips, key=lambda x: x.start_time)
    ]
    return JobOut(
        id=job.id,
        status=job.status,
        status_label=STATUS_LABELS.get(job.status, job.status),
        original_filename=job.original_filename,
        clips_total=job.clips_total,
        clips_done=job.clips_done,
        error=job.error,
        download_ready=(job.status == "complete"),
        created_at=job.created_at,
        updated_at=job.updated_at,
        clips=clip_outs,
    )


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=TokenResponse, status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    if len(req.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters")

    token = secrets.token_urlsafe(32)
    user  = User(
        email                    = req.email,
        password_hash            = hash_password(req.password),
        email_verified           = False,
        email_verification_token = token,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    _send_verification_email(user.email, token)

    return TokenResponse(access_token=create_token(user.id))


@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return TokenResponse(access_token=create_token(user.id))


@app.get("/auth/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


@app.get("/auth/verify-email/{token}")
def verify_email(token: str, db: Session = Depends(get_db)):
    """
    Clicked from the verification email link.
    Marks the user verified, clears the token, redirects to the app.
    """
    user = db.query(User).filter(User.email_verification_token == token).first()
    if not user:
        # Token not found or already used — still redirect, don't leak info
        return RedirectResponse("/?verified=invalid")

    user.email_verified           = True
    user.email_verification_token = None
    user.updated_at               = datetime.utcnow()
    db.commit()

    return RedirectResponse("/?verified=1")


@app.post("/auth/resend-verification", status_code=202)
def resend_verification(
    user: User  = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if user.email_verified:
        raise HTTPException(400, "Email is already verified")

    token                        = secrets.token_urlsafe(32)
    user.email_verification_token = token
    user.updated_at               = datetime.utcnow()
    db.commit()

    _send_verification_email(user.email, token)
    return {"detail": "Verification email sent"}


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/upload/presign", response_model=PresignResponse, status_code=201)
def presign(
    req: PresignRequest,
    user: User  = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # ── Security gate 1: email must be verified ───────────────────────────────
    if not user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your email before uploading.",
        )

    # ── Security gate 2: quota check ──────────────────────────────────────────
    _check_quota(user, db)

    # ── Security gate 3: file size ────────────────────────────────────────────
    if req.size_bytes < 1024:
        raise HTTPException(400, "File is too small to be a valid video (< 1 KB)")
    if req.size_bytes > MAX_UPLOAD_BYTES:
        limit_gb = MAX_UPLOAD_BYTES / (1024 ** 3)
        raise HTTPException(400, f"File exceeds the {limit_gb:.0f} GB upload limit")

    # ── Security gate 4: MIME type whitelist ──────────────────────────────────
    if req.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            400,
            f"File type '{req.content_type}' is not allowed. "
            f"Accepted: {', '.join(sorted(ALLOWED_MIME_TYPES))}",
        )

    # ── Security gate 5: file extension whitelist ─────────────────────────────
    ext = Path(req.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"File extension '{ext}' is not allowed. "
            f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    job = Job(user_id=user.id, original_filename=req.filename, status="uploading")
    db.add(job)
    db.commit()
    db.refresh(job)

    upload_url, gcs_path = generate_upload_url(job.id, req.filename, req.content_type)
    job.gcs_input_path = gcs_path
    db.commit()

    return PresignResponse(job_id=job.id, upload_url=upload_url, gcs_path=gcs_path)


@app.post("/jobs/{job_id}/confirm", status_code=202)
def confirm_upload(
    job_id: str,
    user: User  = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == user.id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "uploading":
        raise HTTPException(400, f"Unexpected job status: {job.status}")

    job.status     = "pending"
    job.updated_at = datetime.utcnow()
    db.commit()

    enqueue_job(job_id)
    return {"job_id": job_id, "status": "pending"}


# ── Jobs ──────────────────────────────────────────────────────────────────────

@app.get("/jobs", response_model=List[JobOut])
def list_jobs(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    jobs = (
        db.query(Job)
        .filter(Job.user_id == user.id)
        .order_by(Job.created_at.desc())
        .limit(50)
        .all()
    )
    return [_job_to_out(j, j.clips) for j in jobs]


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == user.id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    return _job_to_out(job, job.clips)


@app.get("/jobs/{job_id}/download")
def download_job(job_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == user.id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "complete":
        raise HTTPException(400, f"Job is not complete (status: {job.status})")
    if not job.gcs_output_path:
        raise HTTPException(500, "Output not available")
    return RedirectResponse(generate_download_url(job.gcs_output_path))


# ── Billing ───────────────────────────────────────────────────────────────────

@app.get("/billing", response_model=UsageOut)
def billing(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    usage = _get_usage(db, user.id)
    return UsageOut(
        tier=user.tier,
        billing_period=usage.billing_period,
        videos_processed=usage.videos_processed,
        minutes_processed=round(usage.minutes_processed, 1),
        free_videos_remaining=(
            max(0, FREE_VIDEO_LIMIT - usage.videos_processed) if user.tier == "free" else None
        ),
        paid_minutes_remaining=(
            max(0.0, PAID_MINUTE_LIMIT - usage.minutes_processed) if user.tier == "paid" else None
        ),
    )


# ── Internal (local dev only) ─────────────────────────────────────────────────

@app.put("/internal/upload/{job_id}")
async def internal_upload(job_id: str, request: Request, filename: str = "video.mp4"):
    """Receive raw video bytes for local dev. Enforces the same size cap as GCS."""
    content_length = int(request.headers.get("content-length", 0))
    if content_length > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // (1024**3)} GB limit")

    dest = LOCAL_UPLOAD_DIR / job_id
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / filename

    body = await request.body()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File exceeds size limit")

    with open(path, "wb") as f:
        f.write(body)

    db = SessionLocal()
    try:
        db.query(Job).filter(Job.id == job_id).update(
            {"gcs_input_path": f"local://uploads/{job_id}/{filename}"}
        )
        db.commit()
    finally:
        db.close()

    return {"stored": str(path)}


@app.get("/internal/download/{path:path}")
def internal_download(path: str):
    full = LOCAL_OUTPUT_DIR.parent / path
    if not full.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(full))
