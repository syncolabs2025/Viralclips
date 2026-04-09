#!/usr/bin/env bash
# ── ViralClips — Cloud Run GPU Deployment ─────────────────────────────────────
# Deploys the worker as a Cloud Run Job with NVIDIA L4 GPU (us-central1).
# The API server is deployed as a Cloud Run Service (no GPU needed).
#
# Prerequisites:
#   gcloud auth login && gcloud auth configure-docker
#   Enable APIs: Cloud Run, Artifact Registry, Cloud SQL, Redis (Memorystore)
#
# Usage:
#   chmod +x clouddeploy.sh
#   ./clouddeploy.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config — edit these ───────────────────────────────────────────────────────
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"          # L4 GPUs available here
REPO="viralclips"             # Artifact Registry repo name
IMAGE_TAG="$(git rev-parse --short HEAD)"

API_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/api:$IMAGE_TAG"
WORKER_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/worker:$IMAGE_TAG"

# These should already exist — create via Cloud Console or gcloud CLI
DATABASE_URL="postgresql://viralclips:PASSWORD@/viralclips?host=/cloudsql/$PROJECT_ID:$REGION:viralclips"
REDIS_URL="redis://MEMORYSTORE_IP:6379/0"
GCS_BUCKET="your-viralclips-bucket"
OPENAI_API_KEY="sk-..."
SENDGRID_API_KEY="SG...."
JWT_SECRET="$(openssl rand -hex 32)"

# ── Step 1: Create Artifact Registry repo (idempotent) ────────────────────────
echo "▶ Ensuring Artifact Registry repo exists…"
gcloud artifacts repositories create "$REPO" \
    --repository-format=docker \
    --location="$REGION" \
    --quiet 2>/dev/null || true

# ── Step 2: Build + push API image ────────────────────────────────────────────
echo "▶ Building API image…"
docker build -t "$API_IMAGE" ./backend
docker push "$API_IMAGE"

# ── Step 3: Build + push Worker image (GPU) ───────────────────────────────────
echo "▶ Building Worker image (GPU)…"
docker build -t "$WORKER_IMAGE" ./worker
docker push "$WORKER_IMAGE"

# ── Step 4: Deploy API as Cloud Run Service ───────────────────────────────────
echo "▶ Deploying API service…"
gcloud run deploy viralclips-api \
    --image="$API_IMAGE" \
    --region="$REGION" \
    --platform=managed \
    --allow-unauthenticated \
    --cpu=2 \
    --memory=2Gi \
    --min-instances=1 \
    --max-instances=10 \
    --timeout=60 \
    --set-env-vars="DATABASE_URL=$DATABASE_URL,\
REDIS_URL=$REDIS_URL,\
GCS_BUCKET=$GCS_BUCKET,\
OPENAI_API_KEY=$OPENAI_API_KEY,\
SENDGRID_API_KEY=$SENDGRID_API_KEY,\
JWT_SECRET=$JWT_SECRET,\
JWT_ALGORITHM=HS256" \
    --add-cloudsql-instances="$PROJECT_ID:$REGION:viralclips"

API_URL=$(gcloud run services describe viralclips-api \
    --region="$REGION" --format="value(status.url)")
echo "  API URL: $API_URL"

# ── Step 5: Create/update Worker as Cloud Run Job (GPU) ───────────────────────
# Cloud Run Jobs run to completion — one job instance per video.
# GPU: NVIDIA L4, 16 GB VRAM, Ampere (sm_87)
echo "▶ Deploying Worker job (L4 GPU)…"
gcloud run jobs update viralclips-worker \
    --image="$WORKER_IMAGE" \
    --region="$REGION" \
    --cpu=4 \
    --memory=16Gi \
    --task-timeout=3600 \
    --max-retries=2 \
    --parallelism=1 \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --set-env-vars="DATABASE_URL=$DATABASE_URL,\
REDIS_URL=$REDIS_URL,\
GCS_BUCKET=$GCS_BUCKET,\
OPENAI_API_KEY=$OPENAI_API_KEY,\
SENDGRID_API_KEY=$SENDGRID_API_KEY,\
JWT_SECRET=$JWT_SECRET" \
    --add-cloudsql-instances="$PROJECT_ID:$REGION:viralclips" \
    2>/dev/null || \
gcloud run jobs create viralclips-worker \
    --image="$WORKER_IMAGE" \
    --region="$REGION" \
    --cpu=4 \
    --memory=16Gi \
    --task-timeout=3600 \
    --max-retries=2 \
    --parallelism=1 \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --set-env-vars="DATABASE_URL=$DATABASE_URL,\
REDIS_URL=$REDIS_URL,\
GCS_BUCKET=$GCS_BUCKET,\
OPENAI_API_KEY=$OPENAI_API_KEY,\
SENDGRID_API_KEY=$SENDGRID_API_KEY,\
JWT_SECRET=$JWT_SECRET" \
    --add-cloudsql-instances="$PROJECT_ID:$REGION:viralclips"

echo ""
echo "✓ Deployment complete"
echo "  API:    $API_URL"
echo "  Worker: viralclips-worker (triggered per job via RQ)"
echo ""
echo "Next: update API_BASE_URL in your .env to $API_URL"
echo "      run schema.sql against your Cloud SQL instance"
