# ── GPU-enabled worker image ──────────────────────────────────────────────────
# Base: CUDA 12.1 runtime on Ubuntu 22.04
# GPU:  NVIDIA L4 (Cloud Run GPU — Ampere architecture, sm_87)
# CPU fallback: set ENV FORCE_CPU=1 to skip NVENC and use libx264 instead
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
# Install CUDA-enabled PyTorch first (matches CUDA 12.1 on the L4)
# then the rest of the requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.5.1 \
        torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
# Install everything except torch/torchvision (already installed above)
RUN pip install --no-cache-dir \
    $(grep -v '^torch' requirements.txt | grep -v '^torchvision' | grep -v '^#' | grep -v '^$')

# ── App code ──────────────────────────────────────────────────────────────────
COPY worker.py .

# ── Work dirs ─────────────────────────────────────────────────────────────────
RUN mkdir -p /tmp/viralclips/uploads /tmp/viralclips/outputs

CMD ["python", "worker.py"]
