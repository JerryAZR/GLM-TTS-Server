# GLM-TTS FastAPI server image for local / RunPod GPU inference.
#
# Build:
#   docker build -t glm-tts-server:latest .
# Run locally with mock inference:
#   docker run -p 8000:8000 -e GLM_TTS_MOCK_INFERENCE=1 glm-tts-server:latest
# Run locally with real weights:
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/ckpt:/workspace/ckpt \
#     -v /path/to/authorized_keys.json:/workspace/authorized_keys.json:ro \
#     glm-tts-server:latest

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies: git for downloading models, ffmpeg for pydub audio conversion.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the repository into the image.
COPY . /app

# Install the slimmed inference + API requirements.
RUN pip install --no-cache-dir -r api/requirements.txt

# Default environment variables (can be overridden at runtime).
ENV GLM_TTS_MODEL_DIR=/workspace/ckpt
ENV GLM_TTS_VOICES_DIR=/workspace/voices
ENV GLM_TTS_DEVICE=auto
ENV GLM_TTS_DTYPE=float16
ENV GLM_TTS_PORT=8000
ENV GLM_TTS_MOCK_INFERENCE=0
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Use /health so the container is not killed while models are still loading.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/bin/bash", "/app/runpod-start.sh"]
