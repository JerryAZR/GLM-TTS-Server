#!/bin/bash
set -e

# GLM-TTS RunPod startup script.
# Optional model download + uvicorn server launch.

MODEL_DIR="${GLM_TTS_MODEL_DIR:-/workspace/ckpt}"
HF_REPO="${GLM_TTS_HF_REPO:-zai-org/GLM-TTS}"
PORT="${GLM_TTS_PORT:-8000}"

echo "[runpod-start] MODEL_DIR=${MODEL_DIR}"
echo "[runpod-start] HF_REPO=${HF_REPO}"
echo "[runpod-start] PORT=${PORT}"

# If the model directory is missing or empty, try to download weights from HuggingFace.
# Skipped in mock-inference mode, which never loads weights (keeps CI and local
# mock runs from pulling multi-GB checkpoints).
if [ "${GLM_TTS_MOCK_INFERENCE:-0}" = "1" ]; then
    echo "[runpod-start] Mock inference mode; skipping model download."
elif [ ! -d "${MODEL_DIR}" ] || [ -z "$(ls -A "${MODEL_DIR}" 2>/dev/null)" ]; then
    echo "[runpod-start] Model directory is empty. Downloading from ${HF_REPO} ..."
    mkdir -p "${MODEL_DIR}"
    # Note: --local-dir-use-symlinks was removed in huggingface_hub >= 0.26;
    # --local-dir now always writes real files, so no flag is needed.
    huggingface-cli download "${HF_REPO}" --local-dir "${MODEL_DIR}"
    echo "[runpod-start] Model download complete."
else
    echo "[runpod-start] Reusing existing model directory."
fi

# Ensure the voices directory exists and seed it with bundled voices if empty.
VOICES_DIR="${GLM_TTS_VOICES_DIR:-/workspace/voices}"
mkdir -p "${VOICES_DIR}"
if [ -z "$(ls -A "${VOICES_DIR}" 2>/dev/null)" ] && [ -d /app/voices ]; then
    echo "[runpod-start] Seeding voices directory from bundled voices..."
    cp -r /app/voices/* "${VOICES_DIR}/"
fi

cd /app

echo "[runpod-start] Starting GLM-TTS API server on port ${PORT} ..."
exec python -m uvicorn api.server:app --host 0.0.0.0 --port "${PORT}"
