# ViralClips

Upload long-form video → AI extracts viral moments → reframes to 9:16 → zip download.

```
Upload (direct to GCS) → Whisper transcription → GPT-4o mini moment detection
→ content classification → intelligent reframing → zip → email notification
```

**Reframing strategies:**

| Content type | Detection | Method |
|---|---|---|
| `single_face` | MediaPipe Face Detection | Kalman-filtered crop centred on largest face |
| `dual_face` | MediaPipe Face Mesh + lip aperture | Stacked split — active speaker top 60%, listener bottom 40% |
| `no_face` | OpenCV SpectralResidual saliency | EMA-smoothed crop centred on salient region |

---

## Quick start (local dev)

### 1. Clone and configure

```bash
git clone <repo-url> && cd viralclips
cp .env.example .env
# Edit .env — set OPENAI_API_KEY at minimum. Leave GCS_BUCKET empty for local storage.
```

### 2. Run with Docker Compose

```bash
docker compose up --build
```

- API → http://localhost:8000
- Postgres → localhost:5432
- Redis → localhost:6379

Schema is applied automatically on first boot via `docker-entrypoint-initdb.d`.

### 3. Scale workers

```bash
docker compose up --scale worker=4
```

Each worker instance pulls jobs from the Redis queue independently.

---

## Architecture

```
Browser
  │  PUT video (direct to GCS / local API)
  │  POST /jobs/{id}/confirm
  ▼
FastAPI (Cloud Run Service)
  │  enqueue job_id → Redis
  ▼
Redis Queue (RQ)
  │  dequeue
  ▼
Worker (Cloud Run Job, NVIDIA L4)
  ├─ Download video from GCS
  ├─ Whisper API → transcript
  ├─ GPT-4o mini → [start, end, reason, hook_score] × 5
  ├─ For each clip:
  │   ├─ classify_content()  → single_face / dual_face / no_face
  │   ├─ reframe_single_face()  Kalman + face tracking
  │   ├─ reframe_dual_face()   ASD + vstack
  │   └─ reframe_no_face()     saliency + EMA
  ├─ Zip clips → upload to GCS
  └─ SendGrid email notification
```

---

## Project structure

```
viralclips/
├── schema.sql              Postgres DDL (users, jobs, clips, usage_records)
├── docker-compose.yml      Local dev: postgres + redis + api + worker
├── clouddeploy.sh          GCP deployment script
├── backend/
│   ├── main.py             FastAPI — all endpoints
│   ├── auth.py             JWT + bcrypt
│   ├── models.py           SQLAlchemy ORM (Postgres)
│   ├── database.py         Engine + session
│   ├── schemas.py          Pydantic request/response models
│   ├── storage.py          GCS / local storage abstraction
│   ├── Dockerfile
│   ├── requirements.txt
│   └── static/
│       └── index.html      React SPA (CDN React, no build step)
└── worker/
    ├── worker.py           Full AI pipeline (6 pieces)
    ├── Dockerfile          nvidia/cuda:12.1 base, NVENC support
    └── requirements.txt
```

---

## API reference

### Auth
```
POST /auth/register   {email, password}  → {access_token}
POST /auth/login      {email, password}  → {access_token}
GET  /auth/me                            → {id, email, tier}
```

### Upload flow
```
POST /upload/presign  {filename, content_type, size_bytes}
                      → {job_id, upload_url, gcs_path}

# Client PUTs raw video bytes to upload_url (direct to GCS or local endpoint)

POST /jobs/{job_id}/confirm              → {status: "pending"}
```

### Job polling (every 5 s)
```
GET /jobs/{job_id}   → {
  status, status_label,
  clips_total, clips_done,
  clips: [{start_time, end_time, reason, hook_score, content_type, status}],
  download_ready
}
```

Status values: `pending → downloading → transcribing → detecting → reframing → zipping → complete | failed`

### Download
```
GET /jobs/{job_id}/download  → 302 redirect to signed GCS URL (24 h TTL)
```

### Billing
```
GET /billing  → {tier, videos_processed, minutes_processed, free_videos_remaining}
```

---

## Tiers

| | Free | Paid ($10/mo) |
|---|---|---|
| Videos | 3 per period | Unlimited |
| Input length | Up to 1 hour | Up to 1 hour |
| Clips per video | Up to 5 | Up to 5 |
| Output | Watermarked | Clean |
| Email notification | ✓ | ✓ |

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | Postgres connection string |
| `REDIS_URL` | Yes | Redis connection string |
| `OPENAI_API_KEY` | Yes | Whisper + GPT-4o mini |
| `JWT_SECRET` | Yes | Random string for JWT signing |
| `GCS_BUCKET` | Prod only | GCS bucket name — leave empty for local dev |
| `SENDGRID_API_KEY` | Optional | Email notifications |
| `FROM_EMAIL` | Optional | Sender address for notifications |
| `LOCAL_UPLOAD_DIR` | Dev only | Local upload path (default `/tmp/viralclips/uploads`) |
| `LOCAL_OUTPUT_DIR` | Dev only | Local output path (default `/tmp/viralclips/outputs`) |
| `API_BASE_URL` | Dev only | Used to build local upload URLs |

---

## Cloud Run GPU deployment

```bash
# Edit PROJECT_ID, DATABASE_URL, REDIS_URL, GCS_BUCKET in clouddeploy.sh first
chmod +x clouddeploy.sh && ./clouddeploy.sh
```

The script:
1. Creates an Artifact Registry Docker repo
2. Builds + pushes API and Worker images
3. Deploys API as a Cloud Run **Service** (always-on, auto-scales to 10)
4. Deploys Worker as a Cloud Run **Job** (L4 GPU, 16 GB RAM, 1-hour timeout)

Workers are triggered by the RQ queue — they start when there's a job and Cloud Run scales to zero between jobs.

### GCP services needed

| Service | Purpose |
|---|---|
| Cloud Run | API service + GPU worker jobs |
| Artifact Registry | Docker image storage |
| Cloud SQL (Postgres 16) | Job metadata |
| Memorystore (Redis) | RQ job queue |
| Cloud Storage | Video input + clip output |
| Cloud SQL Auth Proxy | Secure DB connections |

---

## Local dev without GCS

When `GCS_BUCKET` is unset:
- Upload URLs point to `POST /internal/upload/{job_id}` on the API server
- Files are stored under `LOCAL_UPLOAD_DIR` / `LOCAL_OUTPUT_DIR`
- Downloads served from `GET /internal/download/{path}`

The frontend and worker code is identical — only the storage layer switches.

---

## Worker packages

| Package | Purpose |
|---|---|
| `openai` | Whisper API + GPT-4o mini |
| `opencv-python-headless` | Video decode, saliency, frame processing |
| `mediapipe` | Face detection, Face Mesh, lip landmarks |
| `light-asd` | Active speaker detection (optional, graceful fallback) |
| `ffmpeg` (system) | Encode 9:16 clips — NVENC on GPU, libx264 on CPU |
| `rq` | Redis-based task queue |
| `psycopg2-binary` | Direct Postgres access (no ORM overhead in worker) |
| `sendgrid` | Email notifications |
| `torch` | Required by MediaPipe |
