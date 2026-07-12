# GLM-TTS API Server

An OpenAI-compatible FastAPI server for [GLM-TTS](https://github.com/zai-org/GLM-TTS). It supports zero-shot voice cloning through separate **voice upload** and **speech generation** endpoints, and is packaged for deployment on RunPod GPU pods.

---

## Features

- OpenAI-style `/v1/audio/speech` TTS endpoint
- Separate `/v1/voices` CRUD endpoints for managing reference voices
- Environment-variable configuration with sensible defaults
- Optional mock-inference mode for testing the API plumbing without model weights
- Bundled `jerry` sample voice

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GLM_TTS_API_KEY` | *(unset)* | If set, all endpoints (except `/health` and `/ready`) require `Authorization: Bearer <key>`. |
| `GLM_TTS_MODEL_DIR` | `/workspace/ckpt` (fallback `ckpt`) | Directory containing the GLM-TTS checkpoints. |
| `GLM_TTS_VOICES_DIR` | `/workspace/voices` (fallback `voices`) | Directory where uploaded voices are stored. |
| `GLM_TTS_DEVICE` | `auto` | `auto`, `cpu`, or `cuda`. |
| `GLM_TTS_DTYPE` | `float16` | `float16`, `bfloat16`, or `float32`. |
| `GLM_TTS_PORT` | `8000` | HTTP port for the API. |
| `GLM_TTS_MOCK_INFERENCE` | `0` | Set to `1` to return a dummy WAV without loading models. |
| `GLM_TTS_SAMPLE_RATE` | `24000` | Output sample rate (24 kHz or 32 kHz). |

---

## Local Install and Run

### 1. Clone the fork

```bash
cd GLM-TTS-Server
```

### 2. Install dependencies

Use the slimmed inference requirements:

```bash
pip install -r api/requirements.txt
```

> If you already installed the full `requirements.txt`, that also works.

### 3. Download model weights (skip if using mock mode)

```bash
mkdir -p ckpt
huggingface-cli download zai-org/GLM-TTS --local-dir ckpt --local-dir-use-symlinks False
```

### 4. Run the server

Mock mode (no weights needed):

```bash
GLM_TTS_MOCK_INFERENCE=1 python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Real inference:

```bash
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

With an API key:

```bash
GLM_TTS_API_KEY=your-secret-key python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

---

## RunPod Deployment

### 1. Build the Docker image

```bash
docker build -t glm-tts-server:latest .
```

### 2. Push to a registry

```bash
docker tag glm-tts-server:latest your-registry/glm-tts-server:latest
docker push your-registry/glm-tts-server:latest
```

### 3. Create a RunPod Pod

1. In the RunPod console, click **Pods** → **Deploy**.
2. Choose a GPU with at least 16 GB VRAM (e.g., RTX 4090 / A5000 / A100).
3. Under **Container Image**, enter your image URL (e.g., `your-registry/glm-tts-server:latest`).
4. Set **Container Port** to `8000` and expose it as **HTTP** (or **TCP** if you prefer).
5. Attach a **Network Volume** and mount it at `/workspace`.
   - On first boot, the image will download `zai-org/GLM-TTS` into `/workspace/ckpt` if it is empty.
   - Uploaded voices will be persisted in `/workspace/voices`.
6. (Optional) set environment variables such as:
   - `GLM_TTS_API_KEY`
   - `GLM_TTS_DTYPE` (default `float16`)
   - `GLM_TTS_DEVICE` (default `auto`)
   - `GLM_TTS_PORT` (default `8000`)

### 4. Verify the pod is ready

```bash
curl https://<runpod-endpoint>/health
curl https://<runpod-endpoint>/ready
```

---

## Example curl Commands

### List models

```bash
curl http://localhost:8000/v1/models
```

### Upload a voice

```bash
curl -X POST http://localhost:8000/v1/voices \
  -F "name=My Voice" \
  -F "prompt_text=Hello, this is my reference voice." \
  -F "prompt_audio=@/path/to/reference.wav" \
  -F "voice_id=my_voice"
```

Response:

```json
{
  "voice_id": "my_voice",
  "name": "My Voice",
  "created_at": "2026-07-12T12:00:00+00:00"
}
```

### List voices

```bash
curl http://localhost:8000/v1/voices
```

### Generate speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-tts",
    "input": "Hello, this is a test of the GLM-TTS API.",
    "voice": "my_voice",
    "response_format": "wav"
  }' \
  --output output.wav
```

For MP3 output:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-tts",
    "input": "你好，这是GLM-TTS的测试。",
    "voice": "my_voice",
    "response_format": "mp3"
  }' \
  --output output.mp3
```

### Authenticated request

If `GLM_TTS_API_KEY` is set, include the header:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-tts","input":"Hello","voice":"my_voice","response_format":"wav"}' \
  --output output.wav
```

---

## Using the Jerry Sample Voice

The fork includes a default sample voice at `voices/jerry/`. The server scans the voices directory on startup and automatically registers it.

### Generate with the sample voice

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-tts",
    "input": "欢迎来到GLM-TTS服务。",
    "voice": "jerry",
    "response_format": "wav"
  }' \
  --output jerry_test.wav
```

### Voice metadata

```bash
curl http://localhost:8000/v1/voices/jerry
```

---

## Docker Run Locally

With mock inference:

```bash
docker run -p 8000:8000 -e GLM_TTS_MOCK_INFERENCE=1 glm-tts-server:latest
```

With real weights mounted from the host:

```bash
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/ckpt:/workspace/ckpt \
  -v $(pwd)/voices:/workspace/voices \
  -e GLM_TTS_API_KEY=your-secret-key \
  glm-tts-server:latest
```

---

## Notes

- The model is kept loaded in GPU memory and only one generation runs at a time (`asyncio.Lock`) to avoid GPU contention.
- Uploaded audio is converted to mono WAV at the configured sample rate (`24 kHz` by default) using `pydub` + `ffmpeg`.
- `response_format` accepts `wav` or `mp3`.
- `speed` is accepted for OpenAI compatibility but currently ignored.
