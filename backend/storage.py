"""
Storage abstraction.

- When GCS_BUCKET is set: use Google Cloud Storage with signed URLs.
- When GCS_BUCKET is unset: use local filesystem (for local dev).

Security enforced here:
- GCS presigned URLs carry X-Goog-Content-Length-Range so GCS itself
  rejects uploads that exceed MAX_UPLOAD_BYTES (10 GB).
- Local dev internal endpoint enforces the same cap via Content-Length header.
"""

import os
from datetime import timedelta
from pathlib import Path

GCS_BUCKET       = os.getenv("GCS_BUCKET", "")
API_BASE_URL     = os.getenv("API_BASE_URL", "http://localhost:8000")
LOCAL_UPLOAD_DIR = Path(os.getenv("LOCAL_UPLOAD_DIR", "/tmp/viralclips/uploads"))
LOCAL_OUTPUT_DIR = Path(os.getenv("LOCAL_OUTPUT_DIR", "/tmp/viralclips/outputs"))

# Hard cap enforced at GCS layer and at our internal upload endpoint.
# 10 GB is generous for up to 1-hour 4K video; adjust down if you want tighter billing control.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024 * 1024)))  # 10 GB


def _gcs_client():
    from google.cloud import storage  # lazy import — not needed in local dev
    return storage.Client()


def generate_upload_url(job_id: str, filename: str, content_type: str) -> tuple[str, str]:
    """
    Returns (upload_url, storage_path).
    Client should PUT the raw video bytes to upload_url.

    GCS rejects the PUT at the storage layer if:
      - Content-Length is outside [1, MAX_UPLOAD_BYTES]   (X-Goog-Content-Length-Range)
      - Content-Type doesn't match what was signed         (built into signed URL)
    """
    if not GCS_BUCKET:
        LOCAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        storage_path = f"local://uploads/{job_id}/{filename}"
        upload_url   = f"{API_BASE_URL}/internal/upload/{job_id}?filename={filename}"
        return upload_url, storage_path

    gcs_path = f"uploads/{job_id}/{filename}"
    client   = _gcs_client()
    blob     = client.bucket(GCS_BUCKET).blob(gcs_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=1),
        method="PUT",
        content_type=content_type,
        # GCS enforces this: upload must be between 1 byte and MAX_UPLOAD_BYTES.
        # Any PUT outside this range is rejected with HTTP 400 before bytes are stored.
        headers={"X-Goog-Content-Length-Range": f"1,{MAX_UPLOAD_BYTES}"},
    )
    return url, f"gs://{GCS_BUCKET}/{gcs_path}"


def generate_download_url(gcs_path: str) -> str:
    """Return a time-limited download URL for a completed zip."""
    if gcs_path.startswith("local://"):
        rel = gcs_path.removeprefix("local://")
        return f"{API_BASE_URL}/internal/download/{rel}"

    blob_name = gcs_path.removeprefix(f"gs://{GCS_BUCKET}/")
    client    = _gcs_client()
    blob      = client.bucket(GCS_BUCKET).blob(blob_name)
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=24),
        method="GET",
    )
