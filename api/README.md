# GLM-TTS API Server

An OpenAI-compatible FastAPI server for [GLM-TTS](https://github.com/zai-org/GLM-TTS). It supports zero-shot voice cloning through separate **voice upload** and **speech generation** endpoints, and is packaged for deployment on RunPod GPU pods.

---

## Features

- OpenAI-style `/v1/audio/speech` TTS endpoint
- Separate `/v1/voices` CRUD endpoints for managing reference voices
- Public-key JWT authentication with pre-enrolled keys (server stores only public keys)
- Legacy single-secret Bearer authentication for local testing
- Environment-variable configuration with sensible defaults
- Optional mock-inference mode for testing the API plumbing without model weights
- Bundled `jerry` sample voice

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GLM_TTS_AUTH_KEYS_FILE` | `/workspace/authorized_keys.json` (or `authorized_keys.json`) | Path to a JSON file of authorized public keys. If present, JWT auth is required. |
| `GLM_TTS_API_KEY` | *(unset)* | Legacy single shared secret. Used only when `GLM_TTS_AUTH_KEYS_FILE` is not configured. |
| `GLM_TTS_MODEL_DIR` | `/workspace/ckpt` (fallback `ckpt`) | Directory containing the GLM-TTS checkpoints. |
| `GLM_TTS_VOICES_DIR` | `/workspace/voices` (fallback `voices`) | Directory where uploaded voices are stored. |
| `GLM_TTS_DEVICE` | `auto` | `auto`, `cpu`, or `cuda`. |
| `GLM_TTS_DTYPE` | `float16` | `float16`, `bfloat16`, or `float32`. |
| `GLM_TTS_PORT` | `8000` | HTTP port for the API. |
| `GLM_TTS_MOCK_INFERENCE` | `0` | Set to `1` to return a dummy WAV without loading models. |
| `GLM_TTS_SAMPLE_RATE` | `24000` | Output sample rate (24 kHz or 32 kHz). |

---

## Authentication

The server supports two authentication modes. They are **mutually exclusive**: if a public-key file is configured, JWT auth is used; otherwise the legacy API key is used. If neither is configured, the server is public.

### Public-key JWT auth (recommended)

The server stores only **public keys** in a JSON file. Clients sign short-lived JWTs with their own private keys and send them in the `Authorization` header.

`authorized_keys.json` format:

```json
{
  "keys": [
    {
      "kid": "tenant-a",
      "name": "Tenant A",
      "public_key": "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...\n-----END PUBLIC KEY-----",
      "scopes": ["speech:generate", "voices:read", "voices:manage"]
    }
  ]
}
```

Supported key algorithms: RSA (`RS256`/`RS384`/`RS512`), ECDSA (`ES256`/`ES384`/`ES512`), Ed25519 (`EdDSA`).

The JWT must include:

- `kid` in the header (matching an entry in the file)
- `sub` claim
- `exp` claim
- `scopes` claim (array of strings)

Available scopes:

| Scope | Endpoint |
|---|---|
| `speech:generate` | `POST /v1/audio/speech` |
| `voices:read` | `GET /v1/voices`, `GET /v1/voices/{id}` |
| `voices:manage` | `POST /v1/voices`, `DELETE /v1/voices/{id}` |

`/v1/models` requires a valid token but no specific scope. `/health` and `/ready` are always public.

### Generating a key pair

```bash
openssl genrsa -out private.pem 2048
openssl rsa -in private.pem -pubout -out public.pem
```

Then add the contents of `public.pem` (newlines preserved) to `authorized_keys.json`.

### Signing a JWT

```bash
python - <<'PY'
import jwt
from datetime import datetime, timezone, timedelta

with open("private.pem", "r") as f:
    private_key = f.read()

now = datetime.now(timezone.utc)
token = jwt.encode(
    {
        "sub": "tenant-a",
        "iat": now,
        "exp": now + timedelta(hours=1),
        "scopes": ["speech:generate"],
    },
    private_key,
    algorithm="RS256",
    headers={"kid": "tenant-a"},
)
print(token)
PY
```

### Legacy single-secret auth

For local testing or simple deployments, you can still use:

```bash
GLM_TTS_API_KEY=your-secret-key python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Clients send:

```http
Authorization: Bearer your-secret-key
```

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

Mock mode (no weights or auth):

```bash
GLM_TTS_MOCK_INFERENCE=1 python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Mock mode with JWT auth:

```bash
GLM_TTS_MOCK_INFERENCE=1 \
GLM_TTS_AUTH_KEYS_FILE=./authorized_keys.json \
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
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
6. Mount or generate an `authorized_keys.json` file at `/workspace/authorized_keys.json`. The server will load it on startup.
7. (Optional) set other environment variables such as:
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

With JWT auth:

```bash
curl -H "Authorization: Bearer $JWT_TOKEN" http://localhost:8000/v1/models
```

Without auth (if no keys configured):

```bash
curl http://localhost:8000/v1/models
```

### Upload a voice

```bash
curl -X POST http://localhost:8000/v1/voices \
  -H "Authorization: Bearer $JWT_TOKEN" \
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
curl -H "Authorization: Bearer $JWT_TOKEN" http://localhost:8000/v1/voices
```

### Generate speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer $JWT_TOKEN" \
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
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-tts",
    "input": "你好，这是GLM-TTS的测试。",
    "voice": "my_voice",
    "response_format": "mp3"
  }' \
  --output output.mp3
```

### Legacy API key request

If `GLM_TTS_API_KEY` is set instead of a public-key file:

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
  -H "Authorization: Bearer $JWT_TOKEN" \
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
curl -H "Authorization: Bearer $JWT_TOKEN" http://localhost:8000/v1/voices/jerry
```

---

## Docker Run Locally

With mock inference and no auth:

```bash
docker run -p 8000:8000 -e GLM_TTS_MOCK_INFERENCE=1 glm-tts-server:latest
```

With real weights and JWT auth mounted from the host:

```bash
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/ckpt:/workspace/ckpt \
  -v $(pwd)/voices:/workspace/voices \
  -v $(pwd)/authorized_keys.json:/workspace/authorized_keys.json:ro \
  glm-tts-server:latest
```

---

## Notes

- The model is kept loaded in GPU memory and only one generation runs at a time (`asyncio.Lock`) to avoid GPU contention.
- Uploaded audio is converted to mono WAV at the configured sample rate (`24 kHz` by default) using `pydub` + `ffmpeg`.
- `/health` returns immediately; `/ready` reflects whether the inference engine has finished loading.
- `response_format` accepts `wav` or `mp3`.
- `speed` is accepted for OpenAI compatibility but currently ignored.
