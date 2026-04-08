FROM python:3.11-slim

# System deps: ffmpeg for video processing, git for whisper model download
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Pre-download the Whisper "base" model (~140 MB) so workers don't fetch it at runtime
RUN python -c "import whisper; whisper.load_model('base')"

# Copy application code
COPY . .

# Create data dirs (overridden by Docker volume mounts)
RUN mkdir -p /data/uploads /data/clips /data/outputs

EXPOSE 8000
