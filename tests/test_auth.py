# JWT public-key authentication tests.
# These tests run in mock mode and use a generated RSA key pair.

import importlib
import json
import os
import time
from datetime import datetime, timezone, timedelta

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient


os.environ["GLM_TTS_MOCK_INFERENCE"] = "1"


def _generate_key_pair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    return public_key, private_key_pem


@pytest.fixture(scope="module")
def key_pair():
    return _generate_key_pair()


@pytest.fixture(scope="module")
def auth_client(key_pair, tmp_path_factory):
    public_key, private_key = key_pair
    # Enroll the key through the real path: a keys file on disk that the
    # server loads during startup (load_authorized_keys clears in-memory
    # state, so enrolling directly before startup would be wiped).
    keys_file = tmp_path_factory.mktemp("auth") / "authorized_keys.json"
    keys_file.write_text(
        json.dumps(
            {
                "keys": [
                    {
                        "kid": "test-key",
                        "name": "Test",
                        "public_key": public_key,
                        "scopes": ["speech:generate", "voices:read", "voices:manage"],
                    }
                ]
            }
        )
    )

    os.environ["GLM_TTS_MOCK_INFERENCE"] = "1"
    os.environ["GLM_TTS_AUTH_KEYS_FILE"] = str(keys_file)

    import api.auth as auth
    import api.server as server

    importlib.reload(auth)  # re-read GLM_TTS_AUTH_KEYS_FILE
    importlib.reload(server)  # re-import AUTH_KEYS_FILE from the reloaded auth

    with TestClient(server.app) as c:
        deadline = time.time() + 10
        while time.time() < deadline:
            resp = c.get("/ready")
            if resp.status_code == 200 and resp.json().get("ready"):
                break
            time.sleep(0.1)
        yield c, private_key


def _make_token(private_key, scopes=None, expired=False, kid="test-key", wrong_key=False):
    now = datetime.now(timezone.utc)
    exp = now - timedelta(hours=1) if expired else now + timedelta(hours=1)
    payload = {
        "sub": "test-user",
        "iat": now,
        "exp": exp,
        "scopes": scopes or ["speech:generate"],
    }
    key = private_key
    if wrong_key:
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        key = other.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()
    return jwt.encode(payload, key, algorithm="RS256", headers={"kid": kid})


def test_models_requires_auth(auth_client):
    client, private_key = auth_client
    resp = client.get("/v1/models")
    assert resp.status_code == 401

    token = _make_token(private_key, scopes=[])
    resp = client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200


def test_generate_speech_with_valid_token(auth_client):
    client, private_key = auth_client
    token = _make_token(private_key, scopes=["speech:generate"])
    resp = client.post(
        "/v1/audio/speech",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": "glm-tts",
            "input": "Hello",
            "voice": "jerry",
            "response_format": "wav",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"


def test_generate_speech_missing_scope(auth_client):
    client, private_key = auth_client
    token = _make_token(private_key, scopes=["voices:read"])
    resp = client.post(
        "/v1/audio/speech",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": "glm-tts",
            "input": "Hello",
            "voice": "jerry",
            "response_format": "wav",
        },
    )
    assert resp.status_code == 403


def test_expired_token(auth_client):
    client, private_key = auth_client
    token = _make_token(private_key, expired=True)
    resp = client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_invalid_signature(auth_client):
    client, private_key = auth_client
    token = _make_token(private_key, wrong_key=True)
    resp = client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_unknown_kid(auth_client):
    client, private_key = auth_client
    token = _make_token(private_key, kid="unknown-key")
    resp = client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401
