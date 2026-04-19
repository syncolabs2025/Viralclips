"""
Modal worker for ViralClips — NVIDIA L4 GPU.

Deployment:
    modal run worker/modal_worker.py          # single job (testing)
    modal deploy worker/modal_worker.py       # persistent RQ worker

The container starts an RQ worker that subscribes to the 'viralclips' Redis
queue.  All models are loaded once in @app.enter() and kept in GPU memory
for the container's lifetime.  Cold-start model loading happens only when
Modal spins up a fresh container.

Environment secrets required (set via: modal secret create viralclips-secrets):
    DATABASE_URL, REDIS_URL, OPENAI_API_KEY, GCS_BUCKET,
    RESEND_API_KEY, FROM_EMAIL, HF_TOKEN (for pyannote diarization)

Model weights are stored in a Modal Volume so they are downloaded once and
reused across container restarts.
"""

import logging
import os
import sys
from pathlib import Path

import modal

log = logging.getLogger("modal_worker")

# ── Modal app ─────────────────────────────────────────────────────────────────

app          = modal.App("viralclips-worker")
model_volume = modal.Volume.from_name("viralclips-models", create_if_missing=True)
MODEL_DIR    = Path("/models")

# ── Container image ───────────────────────────────────────────────────────────

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "ffmpeg", "libsndfile1", "libgl1", "libglib2.0-0",
        "libsm6", "libxrender1", "libxext6",
    ])
    .pip_install([
        # Core ML
        "torch==2.3.1",
        "torchvision==0.18.1",
        "torchaudio==2.3.1",
        "transformers==4.44.0",
        "accelerate==0.33.0",
        "bitsandbytes==0.43.3",      # INT8 quantisation for Qwen2-VL

        # Vision models
        "insightface==0.7.3",
        "onnxruntime-gpu==1.18.1",
        "segment-anything-2",        # SAM-2: pip install git+https://github.com/facebookresearch/sam2
        "groundingdino-py",

        # Audio
        "pyannote.audio==3.3.2",
        "librosa==0.10.2",

        # Shot detection
        "transnetv2",
        "scenedetect[opencv]==0.6.4",

        # CV / encoding
        "opencv-python-headless==4.10.0.84",
        "mediapipe==0.10.14",        # Kalman fallback face detection
        "ffmpeg-python==0.2.0",
        "numpy==1.26.4",
        "scikit-learn==1.5.1",       # DBSCAN for ArcFace clustering
        "Pillow==10.4.0",

        # Pipeline
        "openai==1.40.0",
        "psycopg2-binary==2.9.9",
        "redis==5.0.8",
        "rq==1.16.2",
        "google-cloud-storage==2.18.2",
        "resend==2.3.0",
    ])
)


# ── One-time model download (run manually once) ───────────────────────────────

@app.function(
    image=gpu_image,
    volumes={str(MODEL_DIR): model_volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("viralclips-secrets")],
)
def download_models():
    """
    Run once to populate the model volume:
        modal run worker/modal_worker.py::download_models
    """
    from huggingface_hub import snapshot_download
    import insightface

    log.info("Downloading Qwen2-VL-7B-Instruct …")
    snapshot_download(
        "Qwen/Qwen2-VL-7B-Instruct",
        local_dir=str(MODEL_DIR / "qwen2-vl-7b"),
        ignore_patterns=["*.gguf"],
    )

    log.info("Downloading SAM-2 Large …")
    snapshot_download(
        "facebook/sam2-hiera-large",
        local_dir=str(MODEL_DIR / "sam2"),
    )

    log.info("Downloading pyannote speaker-diarization-3.1 …")
    snapshot_download(
        "pyannote/speaker-diarization-3.1",
        local_dir=str(MODEL_DIR / "pyannote"),
        use_auth_token=os.environ["HF_TOKEN"],
    )

    log.info("Downloading GroundingDINO swin-b …")
    snapshot_download(
        "ShilongLiu/GroundingDINO",
        local_dir=str(MODEL_DIR / "groundingdino"),
    )

    log.info("Downloading Depth-Anything-V2-Large …")
    snapshot_download(
        "depth-anything/Depth-Anything-V2-Large",
        local_dir=str(MODEL_DIR / "depth-anything-v2"),
    )

    log.info("Caching InsightFace buffalo_l …")
    fa = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root=str(MODEL_DIR / "insightface"),
        providers=["CPUExecutionProvider"],
    )
    fa.prepare(ctx_id=-1, det_size=(640, 640))

    model_volume.commit()
    log.info("All models downloaded and committed to volume.")


# ── Worker class (L4 container) ───────────────────────────────────────────────

@app.cls(
    gpu="L4",
    image=gpu_image,
    volumes={str(MODEL_DIR): model_volume},
    container_idle_timeout=600,    # keep warm 10 min between jobs
    timeout=3600,
    secrets=[modal.Secret.from_name("viralclips-secrets")],
    retries=2,
)
class ViralClipsWorker:

    @modal.enter()
    def load_models(self):
        """
        Runs once per container start.  All models loaded into GPU memory here
        and held for the lifetime of the container instance.
        """
        import torch
        from insightface.app import FaceAnalysis
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        device = "cuda"
        log.info("Loading models onto %s …", device)

        # ── InsightFace buffalo_l (face det + ArcFace embeddings) ─────────────
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            root=str(MODEL_DIR / "insightface"),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        log.info("InsightFace ready")

        # ── SAM-2 Large ───────────────────────────────────────────────────────
        try:
            from sam2.build_sam import build_sam2_video_predictor
            self.sam2 = build_sam2_video_predictor(
                str(MODEL_DIR / "sam2" / "sam2_hiera_large.yaml"),
                str(MODEL_DIR / "sam2" / "sam2_hiera_large.pt"),
                device=device,
            )
            log.info("SAM-2 Large ready")
        except Exception as exc:
            log.warning("SAM-2 unavailable (%s) — Kalman fallback active", exc)
            self.sam2 = None

        # ── Qwen2-VL-7B INT8 (local VLM for ambiguous shots) ─────────────────
        try:
            self.vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
                str(MODEL_DIR / "qwen2-vl-7b"),
                torch_dtype=torch.float16,
                load_in_8bit=True,       # ~8 GB VRAM vs 14 GB in FP16
                device_map="cuda",
            )
            self.vl_processor = AutoProcessor.from_pretrained(
                str(MODEL_DIR / "qwen2-vl-7b")
            )
            log.info("Qwen2-VL-7B INT8 ready")
        except Exception as exc:
            log.warning("Qwen2-VL unavailable (%s) — VLM queries disabled", exc)
            self.vl_model     = None
            self.vl_processor = None

        # ── GroundingDINO swin-b ──────────────────────────────────────────────
        try:
            from groundingdino.util.inference import load_model as gdino_load
            cfg_path  = str(MODEL_DIR / "groundingdino" / "GroundingDINO_SwinB_cfg.py")
            ckpt_path = str(MODEL_DIR / "groundingdino" / "groundingdino_swinb_cogcoor.pth")
            self.gdino = gdino_load(cfg_path, ckpt_path)
            log.info("GroundingDINO ready")
        except Exception as exc:
            log.warning("GroundingDINO unavailable (%s) — object detection disabled", exc)
            self.gdino = None

        # ── pyannote speaker diarization ──────────────────────────────────────
        # Loaded lazily inside audio_analysis.run_diarization() to avoid
        # blocking container startup on a slow model download check.

        # ── ReframeEngine (stateless — just holds model refs) ─────────────────
        from reframe.engine import ReframeEngine
        from pathlib import Path
        import tempfile

        self.engine = ReframeEngine(
            face_app=self.face_app,
            sam2=self.sam2,
            vl_model=self.vl_model,
            vl_processor=self.vl_processor,
            gdino=self.gdino,
            work_dir=Path(tempfile.mkdtemp(prefix="reframe_")),
        )

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved()  / 1e9
        log.info("All models loaded. VRAM: %.1f GB allocated / %.1f GB reserved",
                 allocated, reserved)

    @modal.method()
    def run_rq_worker(self):
        """
        Start an RQ worker inside this L4 container.
        Blocks indefinitely, processing jobs from the 'viralclips' queue.
        The models loaded in @enter() stay warm for every job.
        """
        import sys
        from redis import Redis
        from rq import Worker

        # Patch the worker module so its process_job uses our loaded engine
        import worker as worker_module
        worker_module._REFRAME_ENGINE = self.engine

        redis_conn = Redis.from_url(os.environ["REDIS_URL"])
        queues     = sys.argv[1:] or ["viralclips"]
        w          = Worker(queues, connection=redis_conn)
        log.info("RQ worker started — queue(s): %s", queues)
        w.work(with_scheduler=False)


# ── Local entry point ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    """
    Deploy the RQ worker onto a single L4 container:
        modal run worker/modal_worker.py
    """
    ViralClipsWorker().run_rq_worker.remote()
