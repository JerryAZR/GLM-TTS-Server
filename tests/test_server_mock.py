# Mock-mode tests for the GLM-TTS FastAPI server.
# These run against a dedicated mock-mode app from conftest (no weights, no
# GPU, no auth keys -> public access).

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
