# JWT public-key authentication tests against the full server (mock inference).
# Uses a generated RSA key pair enrolled via a real authorized_keys.json.

import json
import time
from datetime import datetime, timezone, timedelta

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient


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


def _wait_ready(c, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = c.get("/ready")
        if resp.status_code == 200 and resp.json().get("ready"):
            return
        time.sleep(0.1)
    raise RuntimeError("Server did not become ready")


def _build_app(keys_file):
    from api.settings import Settings
    from api.server import create_app

    return create_app(Settings(mock_inference=True, auth_keys_file=str(keys_file)))


def _write_keys(path, entries):
    path.write_text(json.dumps({"keys": entries}))
    return path


@pytest.fixture(scope="module")
def auth_client(key_pair, tmp_path_factory):
    public_key, private_key = key_pair
    # Enroll the key through the real path: a keys file on disk that the
    # server loads during startup.
    keys_file = _write_keys(
        tmp_path_factory.mktemp("auth") / "authorized_keys.json",
        [{"public_key": public_key, "role": "admin"}],
    )

    with TestClient(_build_app(keys_file)) as c:
        _wait_ready(c)
        yield c, private_key


def _make_token(private_key, expired=False):
    now = datetime.now(timezone.utc)
    exp = now - timedelta(hours=1) if expired else now + timedelta(hours=1)
    return jwt.encode(
        {"sub": "test-user", "iat": now, "exp": exp},
        private_key,
        algorithm="RS256",
    )


def test_models_requires_auth(auth_client):
    client, private_key = auth_client
    resp = client.get("/v1/models")
    assert resp.status_code == 401

    resp = client.get("/v1/models", headers={"Authorization": f"Bearer {_make_token(private_key)}"})
    assert resp.status_code == 200


def test_generate_speech_with_valid_token(auth_client):
    client, private_key = auth_client
    resp = client.post(
        "/v1/audio/speech",
        headers={"Authorization": f"Bearer {_make_token(private_key)}"},
        json={
            "model": "glm-tts",
            "input": "Hello",
            "voice": "jerry",
            "response_format": "wav",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"


def test_expired_token(auth_client):
    client, private_key = auth_client
    resp = client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {_make_token(private_key, expired=True)}"},
    )
    assert resp.status_code == 401


def test_invalid_signature(auth_client):
    client, _ = auth_client
    other_public, other_private = _generate_key_pair()
    resp = client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {_make_token(other_private)}"},
    )
    assert resp.status_code == 401


def test_user_role_cannot_manage_voices(key_pair, tmp_path):
    """A user-role key can call use-only endpoints but gets 403 on admin ones."""
    public_key, private_key = key_pair
    keys_file = _write_keys(
        tmp_path / "authorized_keys.json",
        [{"public_key": public_key, "role": "user"}],
    )
    with TestClient(_build_app(keys_file)) as client:
        _wait_ready(client)
        headers = {"Authorization": f"Bearer {_make_token(private_key)}"}
        # user role: use-only endpoints work
        assert client.get("/v1/voices", headers=headers).status_code == 200
        assert client.post(
            "/v1/audio/speech",
            headers=headers,
            json={"model": "glm-tts", "input": "Hi", "voice": "jerry"},
        ).status_code == 200
        # admin endpoints reject with 403
        assert client.delete("/v1/voices/jerry", headers=headers).status_code == 403
