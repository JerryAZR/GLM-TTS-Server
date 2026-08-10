# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org).

Release process: bump the app version in `api/server.py`, update this file,
commit, and tag `vX.Y.Z`. Tags are **git-only** — container images are built
per push to `main` (`:latest` / `:sha-<hash>`), so personalized forks always
build from their own pushes.

## [0.1.1] - 2026-08-10

### Fixed

- **Cache poisoning from prompt-feature caching** (introduced in `758938a`):
  `synthesize()` seeded each request's generation cache with the voice's
  *stored* `cache_speech_token` list object. `generate_long` appends chunk
  tokens in place, so the shared list grew across requests while the parallel
  `cache_text`/`cache_text_token` lists were rebuilt per request. Once the
  lists diverged, `get_cached_prompt`'s trim loop popped past the end of
  `cache_text` → `IndexError` → persistent 500s until restart — preceded by a
  window of degraded audio (LLM prompt polluted with prior requests' speech
  tokens) that the degenerate-output retry could not fix, since the corruption
  was in the prompt, not the sampling seed. Fixed by shallow-copying the list
  per request (`c565f80`), with a regression test.

### Changed

- Prompt features are now extracted **eagerly** — at engine load for scanned
  voices and at upload time for new voices — instead of lazily on first
  synthesis (`00d0d83`). First synthesis per voice no longer pays the
  extraction cost, and a reference clip the model cannot process now fails the
  upload immediately (400) instead of on first use (500).

## [0.1.0] - 2026-08-02

Initial release of the serving layer on top of upstream
[zai-org/GLM-TTS](https://github.com/zai-org/GLM-TTS):

- OpenAI-compatible `POST /v1/audio/speech` (wav/mp3), `glm-tts` and
  `glm-tts-mock` models
- Voice management API (`POST`/`GET`/`DELETE /v1/voices`) with per-request
  default-voice resolution and `GLM_TTS_DEFAULT_VOICE`
- Public-key JWT auth: OpenSSH/PEM key enrollment, user/admin roles, open
  mode, `scripts/make_token.py` token minter
- RunPod deployment: Dockerfile, resilient checkpoint download with
  completeness validation, network-volume persistence, bundled-voice sync
- CI/CD: mock-mode test suite, Docker build, GHCR publish with path filtering
- Observability: `/health` `/ready` `/version` `/status` endpoints,
  `scripts/smoke_test.py` one-command deployment check
- Random-seed sampling with degenerate-output retry
  (`X-GLM-TTS-Warning: degenerate-output` header)
- `scripts/setup_wizard.py` for interactive personalization (keys, voices,
  default voice, commit)
