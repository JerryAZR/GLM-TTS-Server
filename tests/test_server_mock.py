# Mock-mode tests for the GLM-TTS FastAPI server.
# These run against a dedicated mock-mode app from conftest (no weights, no
# GPU, no auth keys -> public access).

import io
import wave

import pytest


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True
    assert resp.json()["mock"] is True


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "glm-tts"


def test_list_voices(client):
    resp = client.get("/v1/voices")
    assert resp.status_code == 200
    assert "voices" in resp.json()


def test_version(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert "version" in body
    assert body["mock"] is True


def test_status(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["mock"] is True
    assert body["sample_rate"] == 24000
    assert body["voices"] >= 1
    assert body["default_voice"] == "jerry"  # sole registered voice
    assert body["generating"] is False
    assert body["uptime_seconds"] >= 0
    assert set(body["stats"]) == {
        "speech_requests",
        "failed_requests",
        "audio_seconds_generated",
        "last_generation_seconds",
    }


def test_create_speech_mock(client):
    resp = client.post(
        "/v1/audio/speech",
        json={
            "model": "glm-tts",
            "input": "Hello, this is a mock inference test.",
            "voice": "jerry",
            "response_format": "wav",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert len(resp.content) > 0

    # Stats are updated after a successful generation.
    stats = client.get("/status").json()["stats"]
    assert stats["speech_requests"] >= 1
    assert stats["audio_seconds_generated"] > 0
    assert stats["last_generation_seconds"] is not None


# ---------------------------------------------------------------------------
# Degenerate-output retry
# ---------------------------------------------------------------------------

import torch

from api.server import _is_degenerate_output


def test_is_degenerate_output():
    # The real failure case: 9 words, 0.84s of output.
    text = "Narration drives the entire timeline of the video."
    assert _is_degenerate_output(text, 0.84) is True
    assert _is_degenerate_output(text, 3.12) is False
    # Short utterances never trigger, regardless of duration.
    assert _is_degenerate_output("Hi.", 0.3) is False
    # Chinese: counted per character (12 chars here).
    assert _is_degenerate_output("欢迎来到语音合成服务测试", 1.0) is True
    assert _is_degenerate_output("欢迎来到语音合成服务测试", 3.0) is False


def _fixed_wav(seconds, sample_rate=24000):
    return torch.zeros(1, int(sample_rate * seconds))


def test_retry_recovers_from_degenerate_output(client, app, monkeypatch):
    """Short output triggers a retry with a new seed; the good take wins."""
    engine = app.state.engine
    calls = []

    def fake_synthesize(text, voice, seed):
        calls.append(seed)
        return _fixed_wav(0.1 if len(calls) == 1 else 3.0)

    monkeypatch.setattr(engine, "synthesize", fake_synthesize)
    resp = _speech(client, input="This sentence is long enough to trigger the check.")
    assert resp.status_code == 200
    assert len(calls) == 2  # one degenerate attempt, one good
    assert calls[0] != calls[1]  # fresh random seed per attempt
    assert "x-glm-tts-warning" not in resp.headers


def test_retry_returns_best_with_warning_when_all_degenerate(client, app, monkeypatch):
    engine = app.state.engine
    monkeypatch.setattr(engine, "synthesize", lambda text, voice, seed: _fixed_wav(0.1))
    resp = _speech(client, input="This sentence is long enough to trigger the check.")
    assert resp.status_code == 200
    assert resp.headers.get("x-glm-tts-warning") == "degenerate-output"


# ---------------------------------------------------------------------------
# Prompt-feature caching
# ---------------------------------------------------------------------------


def test_prompt_features_cached_across_requests(client, app, registry, monkeypatch, tmp_path):
    """Prompt-feature extraction runs once per VoiceEntry, not per request."""
    engine = app.state.engine
    voice_dir = tmp_path / "cached_voice"
    voice_dir.mkdir()
    (voice_dir / "prompt_audio.wav").write_bytes(b"RIFF")  # only existence is checked
    registry["cached"] = VoiceEntry(
        voice_id="cached", name="cached", prompt_text="t",
        created_at="2026-01-01T00:00:00+00:00", path=voice_dir,
    )

    calls = []

    def fake_extract(voice):
        calls.append(voice.voice_id)
        return (None, None, None, None, None, None)

    monkeypatch.setattr(engine, "_extract_prompt_features", fake_extract)
    monkeypatch.setattr(engine.settings, "mock_inference", False)
    monkeypatch.setattr(
        "api.server.generate_long", lambda **kw: (torch.zeros(1, 24000), None, None, None)
    )

    assert _speech(client, input="hi", voice="cached").status_code == 200
    assert _speech(client, input="hi", voice="cached").status_code == 200
    assert calls == ["cached"]  # extracted once, reused on the second request


def test_generation_does_not_mutate_cached_features(client, app, registry, monkeypatch, tmp_path):
    """Regression: the per-request generation cache must not alias the cached
    feature list (generate_long appends to it in place, per chunk)."""
    engine = app.state.engine
    voice_dir = tmp_path / "alias_voice"
    voice_dir.mkdir()
    (voice_dir / "prompt_audio.wav").write_bytes(b"RIFF")
    registry["alias"] = VoiceEntry(
        voice_id="alias", name="alias", prompt_text="t",
        created_at="2026-01-01T00:00:00+00:00", path=voice_dir,
    )

    stored_tokens = [[1, 2, 3]]
    monkeypatch.setattr(
        engine, "_extract_prompt_features",
        lambda voice: ("t", [0], stored_tokens, None, None, None),
    )
    monkeypatch.setattr(engine.settings, "mock_inference", False)

    def fake_generate_long(**kw):
        cache = kw["cache"]
        # Simulate generate_long's in-place per-chunk appends.
        cache["cache_text"].append("chunk")
        cache["cache_text_token"].append([9])
        cache["cache_speech_token"].append([4, 5, 6])
        return (torch.zeros(1, 24000), None, None, None)

    monkeypatch.setattr("api.server.generate_long", fake_generate_long)

    assert _speech(client, input="hi", voice="alias").status_code == 200
    assert _speech(client, input="hi", voice="alias").status_code == 200
    assert stored_tokens == [[1, 2, 3]]  # never mutated by either request


# ---------------------------------------------------------------------------
# Voice upload
# ---------------------------------------------------------------------------


def _small_wav_bytes():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * 1600)
    return buf.getvalue()


def test_create_voice_returns_201(client, registry):
    resp = client.post(
        "/v1/voices",
        data={"name": "upload-test", "prompt_text": "hello", "voice_id": "upload_test"},
        files={"prompt_audio": ("prompt.wav", _small_wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 201
    assert "upload_test" in registry


def test_create_voice_rejects_oversized_upload(client, app, registry, monkeypatch):
    monkeypatch.setattr(app.state.settings, "max_upload_bytes", 8)
    resp = client.post(
        "/v1/voices",
        data={"name": "too-big", "prompt_text": "hello", "voice_id": "too_big"},
        files={"prompt_audio": ("prompt.wav", _small_wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 413
    assert "too_big" not in registry


# ---------------------------------------------------------------------------
# Default voice resolution
# ---------------------------------------------------------------------------

from pathlib import Path

from fastapi import HTTPException

from api.server import VoiceEntry, _pick_default, resolve_voice


def _fake_voice(voice_id):
    return VoiceEntry(
        voice_id=voice_id, name=voice_id, prompt_text="t",
        created_at="2026-01-01T00:00:00+00:00", path=Path("/nonexistent"),
    )


@pytest.fixture
def registry(app):
    """A scratch voice registry on the test app, restored after each test."""
    voices = app.state.voices
    saved = dict(voices)
    yield voices
    voices.clear()
    voices.update(saved)


def _speech(client, **kw):
    payload = {"model": "glm-tts", "input": "hi", "response_format": "wav"}
    payload.update(kw)
    return client.post("/v1/audio/speech", json=payload)


# --- pure-function tiers ---------------------------------------------------

def test_resolve_explicit_wins(registry):
    registry.update({"a": _fake_voice("a"), "default": _fake_voice("default")})
    assert resolve_voice("a", registry, "default") == "a"


def test_resolve_explicit_unknown_404(registry):
    with pytest.raises(HTTPException) as e:
        resolve_voice("nope", registry)
    assert e.value.status_code == 404


def test_resolve_configured_default(registry):
    registry.update({"a": _fake_voice("a"), "b": _fake_voice("b")})
    assert resolve_voice(None, registry, "b") == "b"


def test_resolve_dangling_configured_default_is_loud(registry):
    registry.update({"a": _fake_voice("a")})
    with pytest.raises(HTTPException) as e:
        resolve_voice(None, registry, "missing")
    assert e.value.status_code == 400
    assert "GLM_TTS_DEFAULT_VOICE" in e.value.detail


def test_resolve_named_default_beats_sole_voice(registry):
    registry.clear()
    registry.update({"a": _fake_voice("a")})
    assert _pick_default(registry) == "a"  # sole voice
    registry["default"] = _fake_voice("default")
    assert _pick_default(registry) == "default"  # named default wins


def test_resolve_multiple_without_default_fails(registry):
    registry.update({"a": _fake_voice("a"), "b": _fake_voice("b")})
    assert _pick_default(registry) is None
    with pytest.raises(HTTPException) as e:
        resolve_voice(None, registry)
    assert e.value.status_code == 400
    assert "a" in e.value.detail and "b" in e.value.detail


def test_resolve_no_voices_404():
    with pytest.raises(HTTPException) as e:
        resolve_voice(None, {})
    assert e.value.status_code == 404


# --- through the HTTP endpoint --------------------------------------------

def test_speech_omitted_voice_uses_sole_registered(client, registry):
    registry.clear()
    registry["jerry"] = _fake_voice("jerry")
    assert _speech(client).status_code == 200


def test_speech_omitted_voice_multiple_is_400(client, registry):
    registry.clear()
    registry.update({"jerry": _fake_voice("jerry"), "bob": _fake_voice("bob")})
    resp = _speech(client)
    assert resp.status_code == 400
    assert "jerry" in resp.json()["detail"]


def test_speech_omitted_voice_env_default(client, app, registry, monkeypatch):
    registry.clear()
    registry.update({"jerry": _fake_voice("jerry"), "bob": _fake_voice("bob")})
    monkeypatch.setattr(app.state.settings, "default_voice", "bob")
    assert _speech(client).status_code == 200


def test_speech_omitted_voice_dangling_env_default(client, app, monkeypatch):
    monkeypatch.setattr(app.state.settings, "default_voice", "missing")
    resp = _speech(client)
    assert resp.status_code == 400
    assert "missing" in resp.json()["detail"]
