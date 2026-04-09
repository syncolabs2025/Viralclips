from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr


# ── Auth ─────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    id: str
    email: str
    tier: str
    email_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ── Upload / Jobs ─────────────────────────────────────────────────────────────

class PresignRequest(BaseModel):
    filename: str
    content_type: str
    size_bytes: int

class PresignResponse(BaseModel):
    job_id: str
    upload_url: str      # PUT to this URL with the raw video bytes
    gcs_path: str

class ClipOut(BaseModel):
    id: str
    start_time: float
    end_time: float
    duration: float
    reason: str
    hook_score: Optional[int]
    content_type: Optional[str]
    status: str

    class Config:
        from_attributes = True

class JobOut(BaseModel):
    id: str
    status: str
    status_label: str
    original_filename: str
    clips_total: int
    clips_done: int
    error: Optional[str]
    download_ready: bool
    created_at: datetime
    updated_at: datetime
    clips: List[ClipOut] = []

    class Config:
        from_attributes = True


# ── Billing ───────────────────────────────────────────────────────────────────

class UsageOut(BaseModel):
    tier: str
    billing_period: str
    videos_processed: int
    minutes_processed: float
    # Limits
    free_videos_remaining: Optional[int]   # None for paid
    paid_minutes_remaining: Optional[float]


# ── Helpers ───────────────────────────────────────────────────────────────────

STATUS_LABELS = {
    "pending":      "Queued",
    "downloading":  "Downloading video…",
    "transcribing": "Transcribing audio…",
    "detecting":    "Detecting viral moments…",
    "reframing":    "Reframing clips…",
    "zipping":      "Packaging clips…",
    "complete":     "Done",
    "failed":       "Failed",
}
