# ViralClips

Upload any long-form video and get AI-identified highlight clips reframed to 9:16 portrait (Reels / Shorts).

```
Upload video → Whisper transcription → GPT-4o mini moment detection
            → Runway ML 9:16 reframing (parallel) → download zip
```

---

## Quick start (Docker Compose)

### 1. Clone and configure

```bash
git clone <repo-url>
cd viralclips
cp .env.example .env
```

Edit `.env` and set your API keys:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI key — used for Whisper and GPT-4o mini |
| `RUNWAY_API_KEY` | Runway ML key — used for 9:16 reframing |
| `REDIS_URL` | Leave as-is for local Docker Compose |
| `PUBLIC_BASE_URL` | Optional. If set (e.g. `https://api.myapp.com`) Runway fetches clips via HTTP instead of inline base64. Required for very large clips. |

### 2. Build and run

```bash
docker compose up --build
```

Open **http://localhost:8000** in your browser.

### 3. Scale workers

```bash
# Run 4 workers in parallel
docker compose up --scale worker=4
```

Each worker independently picks tasks from Redis — no coordination needed.

---

## Project structure

```
viralclips/
├── docker-compose.yml
├── .env.example
└── backend/
    ├── Dockerfile
    ├── requirements.txt
    ├── main.py          # FastAPI — upload / status / download endpoints
    ├── database.py      # SQLAlchemy engine + session
    ├── models.py        # Job and Clip ORM models
    ├── celery_app.py    # Celery + Redis config
    ├── tasks.py         # Pipeline tasks (transcribe → detect → reframe → zip)
    └── static/
        ├── index.html   # SPA
        └── app.js       # Upload + polling logic
```

---

## API

### `POST /upload`

Upload a video file. Returns immediately with a `job_id`.

```
Content-Type: multipart/form-data
Body: file=<video>

Response 202:
{ "job_id": "uuid", "status": "pending" }
```

### `GET /jobs/{job_id}`

Poll for job progress.

```json
{
  "job_id": "...",
  "status": "reframing",
  "status_label": "Reframing clips to 9:16…",
  "clips_done": 2,
  "clips_total": 5,
  "download_ready": false,
  "clips": [
    {
      "id": "...",
      "start_time": 42.1,
      "end_time": 78.5,
      "duration": 36.4,
      "reason": "High-energy moment with a strong hook.",
      "status": "complete"
    }
  ]
}
```

Job `status` values: `pending → transcribing → detecting → reframing → complete | failed`

### `GET /jobs/{job_id}/download`

Returns the zip file of 9:16 mp4 clips. Only available when `status == "complete"`.

---

## Pipeline detail

### Task 1 — `transcribe_video`
Runs **OpenAI Whisper** (`base` model) locally on the worker. Produces a JSON transcript with per-segment timestamps. Chains to Task 2.

### Task 2 — `detect_viral_moments`
Sends the timestamped transcript to **GPT-4o mini** with a prompt that identifies 3–7 standalone moments (15–60 s each) most likely to perform well as short-form content. Creates a `Clip` row per moment and fans out to Task 3 for each clip.

### Task 3 — `reframe_clip`
For each clip:
1. **FFmpeg** cuts the raw segment from the source video.
2. If the clip is longer than Runway's 10-second limit it is split into ≤10 s chunks.
3. Each chunk is submitted to **Runway ML** `video_to_video` with `ratio: 768:1280` (9:16) and an outpainting prompt.
4. Chunks are downloaded and concatenated back together with FFmpeg.
5. Triggers Task 4 after completion.

### Task 4 — `check_and_finalize`
Idempotent check: once all clips are in a terminal state, zips the successful outputs and marks the job `complete`.

---

## Deployment on Railway

1. Push this repo to GitHub.
2. Create a new Railway project → **Deploy from GitHub repo**.
3. Add a **Redis** plugin (Railway Marketplace).
4. Create **two services** pointing at the same repo:
   - **API**: Start command `uvicorn main:app --host 0.0.0.0 --port $PORT`, root dir `backend/`
   - **Worker**: Start command `celery -A celery_app.celery worker --loglevel=info --concurrency=2`, root dir `backend/`
5. Set environment variables on both services (copy from `.env.example`).
6. Set `PUBLIC_BASE_URL` to the Railway API service URL so Runway can fetch clips over HTTPS.
7. Attach a **Volume** to both services at `/data` so uploads and outputs persist.

> **Tip**: Scale the Worker service to multiple instances via Railway's replicas slider. Each replica handles tasks independently.

---

## Local development (without Docker)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1 — Redis
redis-server

# Terminal 2 — API
uvicorn main:app --reload

# Terminal 3 — Worker
celery -A celery_app.celery worker --loglevel=info
```

---

## Environment variables reference

| Variable | Default | Required |
|---|---|---|
| `OPENAI_API_KEY` | — | Yes |
| `RUNWAY_API_KEY` | — | Yes |
| `REDIS_URL` | `redis://localhost:6379/0` | Yes |
| `DATABASE_URL` | `sqlite:///./viralclips.db` | No |
| `UPLOAD_DIR` | `./uploads` | No |
| `CLIPS_DIR` | `./clips` | No |
| `OUTPUT_DIR` | `./outputs` | No |
| `PUBLIC_BASE_URL` | _(empty — uses base64)_ | No |

---

## Notes

- **Whisper model**: `base` is fast and accurate enough for moment detection. Swap to `small` or `medium` in `tasks.py` for better accuracy at the cost of speed/memory.
- **Runway quota**: Each 9:16 reframe consumes one Runway generation. A 5-clip job = 5 generations (more if clips exceed 10 s each).
- **SQLite vs Postgres**: SQLite with WAL mode works fine for a single-node deployment. For multi-node production, set `DATABASE_URL` to a Postgres connection string — no other code changes needed.
- **GPU workers**: Replace the CPU PyTorch wheels in `requirements.txt` with CUDA wheels to accelerate Whisper significantly.
