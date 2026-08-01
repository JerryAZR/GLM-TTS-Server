# Mock-mode tests for the GLM-TTS FastAPI server.
# These tests run without model weights or a GPU and assume public access.

import os
os.environ["GLM_TTS_MOCK_INFERENCE"] = "1"

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
