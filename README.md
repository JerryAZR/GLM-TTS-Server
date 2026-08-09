# GLM-TTS Server

An OpenAI-compatible API server for **GLM-TTS** — controllable & emotion-expressive
zero-shot TTS with multi-reward reinforcement learning.

This is a fork of [zai-org/GLM-TTS](https://github.com/zai-org/GLM-TTS) that adds a
production serving layer. The model itself is unchanged; for model architecture,
training, and research details see [README_upstream.md](README_upstream.md)
([中文](README_zh.md)).

## What the fork adds

- **OpenAI-compatible speech API** — `POST /v1/audio/speech` (wav/mp3), a drop-in
  for OpenAI TTS clients
- **Zero-shot voice cloning API** — manage reference voices (`POST/GET/DELETE
  /v1/voices`) from a 3-10s clip + transcript
- **Public-key JWT auth** — enroll SSH public keys, mint short-lived tokens locally;
  no shared secrets, role-based access (user/admin), open mode optional
- **Setup wizard** — `python scripts/setup_wizard.py` personalizes keys, voices, and
  the default voice, then commits for you
- **RunPod-ready Docker deployment** — resilient checkpoint download, network-volume
  persistence, CI-built GHCR images (`:latest` + `:sha-<hash>`)
- **Observability** — `/health` `/ready` `/version` `/status` endpoints and a
  one-command smoke test

## Quick start

### RunPod (recommended)

1. Fork this repo, clone your fork, and run `python scripts/setup_wizard.py`
   (enrolls *your* SSH public key, adds *your* voice, sets the default voice, commits).
2. Push — CI builds `ghcr.io/<owner>/<repo>:latest` automatically.
3. Deploy on RunPod (GPU >= 16 GB, port 8000 HTTP, network volume >= 30 GB at
   `/workspace`) — full walkthrough: [RunPod Deployment](api/README.md#runpod-deployment).

### Local (Python)

```bash
pip install -r api/requirements.txt
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Set `GLM_TTS_MOCK_INFERENCE=1` to develop against the API without a GPU or model
weights. Full details: [Local Install and Run](api/README.md#local-install-and-run).

### Verify any deployment

```bash
python scripts/smoke_test.py --endpoint http://localhost:8000
```

## Documentation

- **API reference, authentication, environment variables, deployment** — [api/README.md](api/README.md)
- **Model details, training, upstream quick start** — [README_upstream.md](README_upstream.md)

## Citation & license

If you use the model, please cite the upstream paper (see
[README_upstream.md](README_upstream.md)). License terms follow upstream; see
[LICENSE](LICENSE).
