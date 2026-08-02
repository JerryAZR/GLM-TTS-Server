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
| `GLM_TTS_DTYPE` | `float16` | `float16`, `bfloat16`, or `float32`. Forced to `float32` when the device is CPU. |
| `GLM_TTS_PORT` | `8000` | HTTP port for the API. Changing this requires matching changes to the Docker `EXPOSE`/`HEALTHCHECK` and the exposed RunPod port (all default to 8000). |
| `GLM_TTS_MOCK_INFERENCE` | `0` | Set to `1` to return a dummy WAV without loading models. |
| `GLM_TTS_SAMPLE_RATE` | `24000` | Output sample rate (24 kHz or 32 kHz). |
| `GLM_TTS_USE_PHONEME` | `0` | Set to `1` to enable phoneme input in the text frontend. |
| `GLM_TTS_MAX_UPLOAD_BYTES` | `20000000` | Max reference-audio upload size in bytes (larger uploads are rejected with 413). |
| `GLM_TTS_DEFAULT_VOICE` | *(unset)* | Voice used when a speech request omits `voice`. See [Default Voice Resolution](#default-voice-resolution). |
| `GLM_TTS_HF_REPO` | `zai-org/GLM-TTS` | HuggingFace repo the startup script downloads checkpoints from. |

---

## Authentication

The server uses **public-key JWT authentication** with pre-enrolled keys. If no keys file is configured, the server is public.

The server stores only **public keys** in a JSON file — much like SSH's own `authorized_keys`. Clients sign short-lived JWTs with their private keys; the token only proves possession of a key. What that key may do is decided server-side by its enrolled **role**.

`authorized_keys.json` format:

```json
{
  "keys": [
    {
      "public_key": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... user@host",
      "role": "admin"
    },
    {
      "public_key": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... friend@phone",
      "role": "user"
    }
  ]
}
```

That's the whole format — a public key (OpenSSH one-line, comment included, or PEM) plus a role. `role` defaults to `"user"` when omitted.

Roles:

| Role | Can do |
|---|---|
| `user` | `POST /v1/audio/speech`, `GET /v1/voices`, `GET /v1/models`, `GET /status` |
| `admin` | everything `user` can, plus `POST` and `DELETE /v1/voices` |

Tokens carry no key ID: the server tries each enrolled key until the signature verifies (verification is microseconds; key counts are small). The JWT itself needs only `sub` and `exp` claims. Supported algorithms: RSA (`RS256`/`RS384`/`RS512`), ECDSA (`ES256`/`ES384`/`ES512`), Ed25519 (`EdDSA`).

`/health`, `/ready`, and `/version` are always public.

### Generating a key pair

No special tooling needed — `ssh-keygen` is enough:

```bash
ssh-keygen -t ed25519 -f glm-tts-key -C "glm-tts"
```

Then paste the contents of `glm-tts-key.pub` (the whole line, comment included) as the `public_key` of an entry in `authorized_keys.json`. The comment doubles as the identity shown in server logs when the key authenticates. You can also reuse an existing SSH key (`~/.ssh/id_ed25519.pub`), though a dedicated key is tidier. PEM keys generated with `openssl` work too.

### Signing a JWT

Use the bundled helper (installs nothing beyond `PyJWT` + `cryptography`):

```bash
python scripts/make_token.py --key ~/.ssh/glm-tts-key
```

It prints a token good for one hour (see `--expires`, `--sub` for options). The algorithm is inferred from the key type (Ed25519 → `EdDSA`, RSA → `RS256`, EC → `ES256`); encrypted keys prompt for a passphrase.

---

## Local Install and Run

### 1. Clone the fork

```bash
git clone https://github.com/JerryAZR/GLM-TTS-Server.git
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

Mock mode (no weights). Note this repo ships an `authorized_keys.json`, so JWT auth is still active; to run fully open, point `GLM_TTS_AUTH_KEYS_FILE` at a nonexistent path:

```bash
GLM_TTS_MOCK_INFERENCE=1 \
GLM_TTS_AUTH_KEYS_FILE=/nonexistent \
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Mock mode with JWT auth (uses the repo's `authorized_keys.json` by default):

```bash
GLM_TTS_MOCK_INFERENCE=1 \
GLM_TTS_AUTH_KEYS_FILE=./authorized_keys.json \
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

---

## RunPod Deployment

> **Client gotcha:** RunPod's HTTP proxy sits behind Cloudflare bot protection. Non-browser clients whose default `User-Agent` is blocked (e.g. Python `urllib`/`requests`) get `403 error code: 1010`. Set any browser-like `User-Agent` header in your client and requests pass normally. `curl` is not affected. Also note `curl -d` on Windows mangles non-ASCII (e.g. Chinese) text — send UTF-8 JSON from a real client instead.

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
   - GHCR packages pushed by CI are **private by default**: either flip the package to public (GitHub → Packages → `glm-tts-server` → Package settings → Change visibility), or add your GHCR credentials under **Registry Credentials** in the pod template. Pulling fails silently otherwise.
4. Set **Container Port** to `8000` and expose it as **HTTP** (or **TCP** if you prefer).
5. Attach a **Network Volume** of **at least 30 GB** and mount it at `/workspace`.
   - The default 5 GB container disk is **too small**: the model download is multi-GB, and without a volume neither checkpoints nor voices persist across pod stops.
   - On first boot, the image will download `zai-org/GLM-TTS` into `/workspace/ckpt`. Completeness is validated on every boot (`.download-complete` marker, falling back to checking the actual model files): a partial download is wiped and retried, while a complete checkpoint copied in by other means is adopted as-is.
   - The bundled `jerry` sample voice is copied into `/workspace/voices` on first boot if that directory is empty; uploaded voices are persisted there too.
6. (Optional) set other environment variables such as:
   - `GLM_TTS_DTYPE` (default `float16`)
   - `GLM_TTS_DEVICE` (default `auto`)
   - `GLM_TTS_PORT` (default `8000`)

### 4. Enroll your public key

The server loads authorized public keys at startup. It checks `/workspace/authorized_keys.json` first (network volume), then falls back to `authorized_keys.json` in the image (`/app`). Since only **public** keys are involved, committing the file to the repo is safe and is the zero-maintenance option:

**Option A — bake it into the image (recommended).** Generate a key locally (`ssh-keygen -t ed25519 -f glm-tts-key -C "glm-tts"`), create `authorized_keys.json` in the repo root (see [Authentication](#authentication) and `authorized_keys.example.json`), and commit it. The file is copied into the image at `/app/authorized_keys.json` and loaded from the very first boot — the server is never public, and there is nothing to configure per deploy.
   - Only your **public** key ships in the image. Anyone can pull and run the image, but only the holder of the matching private key can authenticate — including to their own deployment, so keep the private key safe.
   - The server **fails to start** if a keys file exists but cannot be parsed (fail-closed); it is only public when no keys file exists at all.

**Option B — place it on the network volume.** Useful for rotating keys without rebuilding the image. Open the pod's **web terminal** (or SSH in) and write the file, e.g.:
   ```bash
   cat > /workspace/authorized_keys.json <<'EOF'
   { "keys": [ { "public_key": "ssh-ed25519 AAAA... glm-tts", "role": "admin" } ] }
   EOF
   ```
Then **restart the pod** (the model is cached on the volume, so the restart is quick). A file at `/workspace/authorized_keys.json` takes precedence over the baked-in one.

> If neither file exists, the server is **public** — enroll via one of the options above before exposing the port.

### 5. Verify the pod (smoke test)

One command checks the whole deployment — liveness, build revision, model readiness, auth enforcement, status, and a real synthesis saved to `smoke_test.wav`:

```bash
python scripts/smoke_test.py --endpoint https://<runpod-endpoint>
```

It prints `[ok]`/`[FAIL]` per check and exits non-zero on failure. Your SSH key is auto-detected (`~/.ssh/id_ed25519`, then `~/.ssh/id_rsa`; pass `--key` to use a different one) — and if the server is public, the token is simply ignored. By default it synthesizes two acronym-free samples, `smoke_test_en.wav` and `smoke_test_zh.wav` (`--lang en|zh|both`, or `--text` for a custom single sample; `--output` sets the file stem). English passing while Chinese fails points at a text encoding problem, not the model. First boot needs patience: the model download is several GB, and `--timeout` defaults to 300s of `/ready` polling. If it ends with `SMOKE TEST PASSED`, play the WAVs — the pod is fully operational.

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

With mock inference (JWT auth still active via the baked-in `authorized_keys.json`; add `-e GLM_TTS_AUTH_KEYS_FILE=/nonexistent` to run open):

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

A manually downloaded `ckpt` (see step 3 above) has no `.download-complete` marker — that's fine: the startup script validates the checkpoint's contents and adopts it, and only wipes directories that look genuinely incomplete.

---

## Default Voice Resolution

`voice` is optional in `POST /v1/audio/speech`. When omitted, the voice is resolved **per request** (the registry changes at runtime via the voices API) in this order:

1. **`voice` in the request** — always wins; unknown name → `404`.
2. **`GLM_TTS_DEFAULT_VOICE`** — operator override. If set but the voice doesn't exist, requests fail loudly with `400` naming the dangling config (never a silent fallback to a different voice); a startup warning is logged too.
3. **A voice with `voice_id` = `"default"`** — convention over config. Manageable entirely through the voices API (upload/rename/delete), no redeploy needed.
4. **Exactly one voice registered** — zero-config convenience.
5. **Otherwise** — `400` listing the available voices.

The resolved value is visible in `GET /status` as `"default_voice"` (`null` when ambiguous).

---

## Observability

- **Interactive API docs** are auto-generated by FastAPI: open `/docs` (Swagger UI — click **Authorize**, paste a token from `make_token.py`, and try endpoints interactively) or `/redoc` in a browser. `/openapi.json` is available for client generators.
- **`GET /version`** (public) — build stamp of the running image: `{"version": "<git-sha>", "mock": false}`. CI-built images carry their commit sha; local builds report `"unknown"`.
- **`GET /status`** (token required, no scope) — quick operational check: readiness, device/dtype/sample rate, uptime, voice count, whether a generation is in progress, and in-memory request stats (speech requests, failures, audio seconds generated, last generation time). Stats reset on restart; anything deeper belongs in an SSH session on the pod.

---

## Pronunciation Hints (Not Supported)

The server has **no pronunciation-hint mechanism** — no SSML, no phoneme tags (the OpenAI speech API it mirrors doesn't have one either). The model guesses pronunciations from context, and it will guess wrong for:

- **Polyphonic characters (多音字)** — e.g. `行` may be read *háng* or *xíng* depending on context
- **Made-up tech words and acronyms** — e.g. "GLM-TTS", "NPC" may be "pronounced" as garbage words
- **Numbers, symbols, mixed-language text** — reading varies by model

**Sanitize the input client-side before sending:** replace ambiguous characters with unambiguous homophones (`这一行(hang2)代码` → `这一航代码`), respell made-up words phonetically, and spell out or drop acronyms. This preprocessing works universally across any TTS backend and keeps full control at the authoring site, where the intended reading is known. The smoke test's own default texts follow this rule (acronym-free by design).

> Under the hood, GLM-TTS does have a pinyin-phoneme pathway that could support exact inline hints (e.g. `行(hang2)`) in the future, but the server does not expose it. Open an issue if a client-side sanitizer proves insufficient.

---

## Notes

- The model is kept loaded in GPU memory and only one generation runs at a time (`asyncio.Lock`) to avoid GPU contention.
- Uploaded audio is converted to mono WAV at the configured sample rate (`24 kHz` by default) using `pydub` + `ffmpeg`.
- `/health` returns immediately; `/ready` reflects whether the inference engine has finished loading.
- `response_format` accepts `wav` or `mp3`.
- `speed` is accepted for OpenAI compatibility but currently ignored.
