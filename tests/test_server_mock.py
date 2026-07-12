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
