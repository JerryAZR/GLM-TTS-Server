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

# If the model directory is missing or incomplete, download weights from
# HuggingFace. Skipped in mock-inference mode, which never loads weights
# (keeps CI and local mock runs from pulling multi-GB checkpoints).
#
# A .download-complete marker guards against reusing a partial download:
# without it, a crashed/killed first download (e.g. disk full) would leave a
# non-empty but broken directory that every later boot would "reuse".
# Checkpoints populated outside this script (copied from another pod, manual
# huggingface-cli download) have no marker, so completeness is validated by
# content instead of wiping blindly.
MARKER="${MODEL_DIR}/.download-complete"

ckpt_complete() {
    [ -f "${MARKER}" ] && return 0
    [ -f "${MODEL_DIR}/flow/flow.pt" ] || return 1
    [ -f "${MODEL_DIR}/flow/config.yaml" ] || return 1
    compgen -G "${MODEL_DIR}/speech_tokenizer/model*.safetensors" > /dev/null || return 1
    [ -n "$(ls -A "${MODEL_DIR}/llm" 2>/dev/null)" ] || return 1
    return 0
}

if [ "${GLM_TTS_MOCK_INFERENCE:-0}" = "1" ]; then
    echo "[runpod-start] Mock inference mode; skipping model download."
elif ckpt_complete; then
    if [ ! -f "${MARKER}" ]; then
        echo "[runpod-start] Found a complete checkpoint without marker; adopting it."
        touch "${MARKER}"
    fi
    echo "[runpod-start] Reusing existing model directory."
else
    if [ -d "${MODEL_DIR}" ] && [ -n "$(ls -A "${MODEL_DIR}" 2>/dev/null)" ]; then
        echo "[runpod-start] WARNING: ${MODEL_DIR} exists but looks incomplete"
        echo "[runpod-start] (interrupted download?). Wiping and re-downloading."
        rm -rf "${MODEL_DIR:?}"
    fi
    echo "[runpod-start] Downloading model from ${HF_REPO} ..."
    mkdir -p "${MODEL_DIR}"
    # Note: --local-dir-use-symlinks was removed in huggingface_hub >= 0.26;
    # --local-dir now always writes real files, so no flag is needed.
    huggingface-cli download "${HF_REPO}" --local-dir "${MODEL_DIR}"
    touch "${MARKER}"
    echo "[runpod-start] Model download complete."
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
