"""
Storage abstraction.

- When GCS_BUCKET is set: use Google Cloud Storage with signed URLs.
- When GCS_BUCKET is unset: use local filesystem (for local dev).
"""

import os
from datetime import timedelta
from pathlib import Path

GCS_BUCKET  = os.getenv("GCS_BUCKET", "")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
LOCAL_UPLOAD_DIR = Path(os.getenv("LOCAL_UPLOAD_DIR", "/tmp/viralclips/uploads"))
LOCAL_OUTPUT_DIR = Path(os.getenv("LOCAL_OUTPUT_DIR", "/tmp/viralclips/outputs"))


def _gcs_client():
    from google.cloud import storage  # lazy import — not needed in local dev
    return storage.Client()


def generate_upload_url(job_id: str, filename: str, content_type: str) -> tuple[str, str]:
    """
    Returns (upload_url, storage_path).
    Client should PUT the raw video bytes to upload_url.
    """
    if not GCS_BUCKET:
        # Local dev: return an endpoint on our own API server
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
    )
    return url, f"gs://{GCS_BUCKET}/{gcs_path}"


def generate_download_url(gcs_path: str) -> str:
    """Return a time-limited download URL for a completed zip."""
    if gcs_path.startswith("local://"):
        # Strip the local:// prefix and serve via API
        rel = gcs_path.removeprefix("local://")
        return f"{API_BASE_URL}/internal/download/{rel}"

    # Strip gs://bucket/ prefix to get the blob name
    blob_name = gcs_path.removeprefix(f"gs://{GCS_BUCKET}/")
    client    = _gcs_client()
    blob      = client.bucket(GCS_BUCKET).blob(blob_name)
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=24),
        method="GET",
    )
