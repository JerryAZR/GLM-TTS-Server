# GLM-TTS API Server

An OpenAI-compatible FastAPI server for [GLM-TTS](https://github.com/zai-org/GLM-TTS). It supports zero-shot voice cloning through separate **voice upload** and **speech generation** endpoints, and is packaged for deployment on RunPod GPU pods.

---

## Features

- OpenAI-style `/v1/audio/speech` TTS endpoint
- Separate `/v1/voices` CRUD endpoints for managing reference voices
- Public-key JWT authentication with pre-enrolled keys (server stores only public keys; accepts OpenSSH `ssh-keygen` keys directly)
- Environment-variable configuration with sensible defaults
- Optional mock-inference mode for testing the API plumbing without model weights
- Bundled `jerry` sample voice

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GLM_TTS_AUTH_KEYS_FILE` | `/workspace/authorized_keys.json` (or `authorized_keys.json`) | Path to a JSON file of authorized public keys. If present, JWT auth is required. |
| `GLM_TTS_MODEL_DIR` | `/workspace/ckpt` (fallback `ckpt`) | Directory containing the GLM-TTS checkpoints. |
| `GLM_TTS_VOICES_DIR` | `/workspace/voices` (fallback `voices`) | Directory where uploaded voices are stored. |
| `GLM_TTS_DEVICE` | `auto` | `auto`, `cpu`, or `cuda`. |
| `GLM_TTS_DTYPE` | `float16` | `float16`, `bfloat16`, or `float32`. |
| `GLM_TTS_PORT` | `8000` | HTTP port for the API. |
| `GLM_TTS_MOCK_INFERENCE` | `0` | Set to `1` to return a dummy WAV without loading models. |
| `GLM_TTS_SAMPLE_RATE` | `24000` | Output sample rate (24 kHz or 32 kHz). |

---

## Authentication

The server uses **public-key JWT authentication** with pre-enrolled keys. If no keys file is configured, the server is public.

The server stores only **public keys** in a JSON file. Clients sign short-lived JWTs with their own private keys and send them in the `Authorization` header.

`authorized_keys.json` format:

```json
{
  "keys": [
    {
      "kid": "my-laptop",
      "name": "My laptop",
      "public_key": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... user@host",
      "scopes": ["speech:generate", "voices:read", "voices:manage"]
    }
  ]
}
```

Public keys may be single-line **OpenSSH format** (`ssh-ed25519 ...`, `ssh-rsa ...`, trailing comments allowed — paste your `.pub` line as-is) or **PEM** (`-----BEGIN PUBLIC KEY-----`). See `authorized_keys.example.json`.

Supported JWT algorithms: RSA (`RS256`/`RS384`/`RS512`), ECDSA (`ES256`/`ES384`/`ES512`), Ed25519 (`EdDSA`).

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

No special tooling needed — `ssh-keygen` is enough:

```bash
ssh-keygen -t ed25519 -f glm-tts-key -C "glm-tts"
```

Then paste the contents of `glm-tts-key.pub` (the whole line, comment included) as the `public_key` of an entry in `authorized_keys.json`. You can also reuse an existing SSH key (`~/.ssh/id_ed25519.pub`), though a dedicated key is tidier. PEM keys generated with `openssl` work too.

### Signing a JWT

Use the bundled helper (installs nothing beyond the server's own `PyJWT` + `cryptography` deps):

```bash
python scripts/make_token.py --key ~/.ssh/glm-tts-key --kid my-laptop
```

It prints a token good for one hour (see `--expires`, `--scopes`, `--sub` for options). The algorithm is inferred from the key type (Ed25519 → `EdDSA`, RSA → `RS256`, EC → `ES256`); encrypted keys prompt for a passphrase.

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
huggingface-cli download zai-org/GLM-TTS --local-dir ckpt
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

CI already does this: every push to `main` publishes `ghcr.io/<owner>/<repo>:latest` (see the `push-image` job in `.github/workflows/ci.yml`). To push manually instead:

```bash
docker tag glm-tts-server:latest your-registry/glm-tts-server:latest
docker push your-registry/glm-tts-server:latest
```

### 3. Create a RunPod Pod

1. In the RunPod console, click **Pods** → **Deploy**.
2. Choose a GPU with at least 16 GB VRAM (e.g., RTX 4090 / A5000 / A100).
3. Under **Container Image**, enter your image URL (e.g., `ghcr.io/<owner>/<repo>:latest`).
4. Set **Container Port** to `8000` and expose it as **HTTP** (or **TCP** if you prefer).
5. Attach a **Network Volume** and mount it at `/workspace`.
   - On first boot, the image will download `zai-org/GLM-TTS` into `/workspace/ckpt` if it is empty.
   - The bundled `jerry` sample voice is copied into `/workspace/voices` on first boot; uploaded voices are persisted there too.
6. (Optional) set other environment variables such as:
   - `GLM_TTS_DTYPE` (default `float16`)
   - `GLM_TTS_DEVICE` (default `auto`)
   - `GLM_TTS_PORT` (default `8000`)

### 4. Enroll your public key

The server loads authorized public keys at startup. It checks `/workspace/authorized_keys.json` first (network volume), then falls back to `authorized_keys.json` in the image (`/app`). Since only **public** keys are involved, committing the file to the repo is safe and is the zero-maintenance option:

**Option A — bake it into the image (recommended).** Generate a key locally (`ssh-keygen -t ed25519 -f glm-tts-key -C "glm-tts"`), create `authorized_keys.json` in the repo root (see [Authentication](#authentication) and `authorized_keys.example.json`), and commit it. The file is copied into the image at `/app/authorized_keys.json` and loaded from the very first boot — the server is never public, and there is nothing to configure per deploy.

**Option B — place it on the network volume.** Useful for rotating keys without rebuilding the image. Open the pod's **web terminal** (or SSH in) and write the file, e.g.:
   ```bash
   cat > /workspace/authorized_keys.json <<'EOF'
   { "keys": [ { "kid": "my-laptop", "public_key": "ssh-ed25519 AAAA... glm-tts", "scopes": ["speech:generate", "voices:read", "voices:manage"] } ] }
   EOF
   ```
Then **restart the pod** (the model is cached on the volume, so the restart is quick). A file at `/workspace/authorized_keys.json` takes precedence over the baked-in one.

> If neither file exists, the server is **public** — enroll via one of the options above before exposing the port.

### 5. Verify the pod (first-deploy smoke test)

Run through this checklist once after deploying:

1. **Model download** — pod logs show `[runpod-start] Model download complete.` (first boot only; several GB, may take a while).
2. **Health/readiness:**
   ```bash
   curl https://<runpod-endpoint>/health
   curl https://<runpod-endpoint>/ready   # {"ready": true} once models are loaded
   ```
3. **Auth is enforced** — without a token you get 401:
   ```bash
   curl -i https://<runpod-endpoint>/v1/models   # HTTP/1.1 401
   ```
4. **Real synthesis** with the bundled voice:
   ```bash
   TOKEN=$(python scripts/make_token.py --key ~/.ssh/glm-tts-key --kid my-laptop)
   curl -X POST https://<runpod-endpoint>/v1/audio/speech \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"model":"glm-tts","input":"你好，这是GLM-TTS的测试。","voice":"jerry","response_format":"wav"}' \
     --output smoke_test.wav
   ```
   Play `smoke_test.wav` — if it speaks, the pod is fully operational.

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

---

## Using the Jerry Sample Voice

The fork includes a default sample voice at `voices/jerry/`. Locally, the server scans the voices directory on startup and automatically registers it. On RunPod, the startup script copies the bundled voices into `/workspace/voices` on first boot (only if that directory is empty), so the voice persists on your network volume alongside any voices you upload later via `POST /v1/voices`.

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
