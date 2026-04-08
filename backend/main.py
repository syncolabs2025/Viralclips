"""
ViralClips — FastAPI backend

Endpoints:
  POST /upload              Upload a video; returns {job_id}
  GET  /jobs/{job_id}       Poll job status + clip metadata
  GET  /jobs/{job_id}/download  Download finished zip
  GET  /health              Liveness check
  GET  /clips/{filename}    Serve extracted clip segments (used by Runway when PUBLIC_BASE_URL is set)
"""

import os
import shutil
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from database import SessionLocal, engine
from models import Base, Clip, Job
from tasks import transcribe_video

# Bootstrap database
Base.metadata.create_all(bind=engine)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
CLIPS_DIR = Path(os.getenv("CLIPS_DIR", "./clips"))

for _d in (UPLOAD_DIR, OUTPUT_DIR, CLIPS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ViralClips", version="1.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve extracted clips (needed when Runway fetches them via PUBLIC_BASE_URL)
app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR)), name="clips")

# Serve the SPA
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", status_code=202)
async def upload_video(file: UploadFile = File(...)):
    """
    Accept a video upload, persist it, record a Job row, and queue the
    transcription task.  Returns the job_id immediately; client should poll
    GET /jobs/{job_id} for progress.
    """
    content_type = file.content_type or ""
    if not content_type.startswith("video/") and not file.filename.lower().endswith(
        (".mp4", ".mov", ".avi", ".mkv", ".webm")
    ):
        raise HTTPException(status_code=400, detail="Only video files are accepted")

    job_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix.lower() or ".mp4"
    video_path = UPLOAD_DIR / f"{job_id}{suffix}"

    with open(video_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    db = SessionLocal()
    try:
        job = Job(
            id=job_id,
            original_filename=file.filename,
            video_path=str(video_path),
            status="pending",
        )
        db.add(job)
        db.commit()
    finally:
        db.close()

    transcribe_video.delay(job_id)

    return {"job_id": job_id, "status": "pending"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    """Poll job status, progress details, and clip metadata."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    clips = db.query(Clip).filter(Clip.job_id == job_id).order_by(Clip.start_time).all()

    status_label = {
        "pending": "Queued",
        "transcribing": "Transcribing audio…",
        "detecting": "Detecting viral moments…",
        "reframing": "Reframing clips to 9:16…",
        "complete": "Done",
        "failed": "Failed",
    }.get(job.status, job.status)

    clips_done = sum(1 for c in clips if c.status == "complete")
    clips_total = len(clips)

    return {
        "job_id": job.id,
        "status": job.status,
        "status_label": status_label,
        "original_filename": job.original_filename,
        "error": job.error,
        "clips_done": clips_done,
        "clips_total": clips_total,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "download_ready": job.status == "complete",
        "clips": [
            {
                "id": c.id,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "duration": round(c.end_time - c.start_time, 1),
                "reason": c.reason,
                "status": c.status,
            }
            for c in clips
        ],
    }


@app.get("/jobs/{job_id}/download")
def download_clips(job_id: str, db: Session = Depends(get_db)):
    """Stream the finished zip file of 9:16 clips."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete yet (status: {job.status})",
        )

    zip_path = OUTPUT_DIR / f"{job_id}_clips.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=500, detail="Output zip not found on disk")

    short_id = job_id[:8]
    safe_name = "".join(
        c if c.isalnum() or c in "-_." else "_"
        for c in Path(job.original_filename).stem
    )
    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename=f"viralclips_{safe_name}_{short_id}.zip",
    )
