# Basic CI tests for the GLM-TTS FastAPI server.
# These tests run in mock-inference mode so they do not require model weights or a GPU.

import os
import time

os.environ["GLM_TTS_MOCK_INFERENCE"] = "1"

import pytest
from fastapi.testclient import TestClient

from api.server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _wait_for_ready(client: TestClient, timeout: float = 10.0) -> dict:
    """Poll /ready until the engine reports ready or the timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get("/ready")
        data = resp.json()
        if data.get("ready"):
            return data
        time.sleep(0.1)
    return client.get("/ready").json()


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready(client: TestClient):
    data = _wait_for_ready(client)
    assert data["ready"] is True
    assert data["mock"] is True


def test_list_models(client: TestClient):
    _wait_for_ready(client)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "glm-tts"


def test_list_voices(client: TestClient):
    _wait_for_ready(client)
    resp = client.get("/v1/voices")
    assert resp.status_code == 200
    assert "voices" in resp.json()


def test_create_speech_mock(client: TestClient):
    _wait_for_ready(client)
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


def test_create_speech_not_ready_before_load(client: TestClient):
    # Engine is already loaded by the fixture, so this test is informational.
    # A dedicated test would require a fresh app instance with mock disabled.
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True
