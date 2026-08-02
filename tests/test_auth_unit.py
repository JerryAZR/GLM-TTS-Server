# Unit tests for api/auth.py.
# These tests do not import the heavy TTS model stack; they exercise the JWT
# public-key verification logic against an AuthState and a minimal FastAPI app.

import json
from datetime import datetime, timezone, timedelta

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.auth import AuthState, load_authorized_keys, _verify_jwt, verify_auth


_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PUBLIC_KEY = _private_key.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
).decode()
PRIVATE_KEY = _private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode()


def _make_token(scopes=None, expired=False, kid="test-key", wrong_key=False):
    now = datetime.now(timezone.utc)
    exp = now - timedelta(hours=1) if expired else now + timedelta(hours=1)
    payload = {
        "sub": "test-user",
        "iat": now,
        "exp": exp,
        "scopes": scopes or ["speech:generate"],
    }
    key = PRIVATE_KEY
    if wrong_key:
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        key = other.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()
    return jwt.encode(payload, key, algorithm="RS256", headers={"kid": kid})


@pytest.fixture
def state():
    """A fresh AuthState per test — no global state to reset."""
    return AuthState()


@pytest.fixture
def key_file(tmp_path):
    path = tmp_path / "authorized_keys.json"
    path.write_text(
        json.dumps(
            {
                "keys": [
                    {
                        "kid": "test-key",
                        "name": "Test",
                        "public_key": PUBLIC_KEY,
                        "scopes": ["speech:generate", "voices:read", "voices:manage"],
                    }
                ]
            }
        )
    )
    return path


@pytest.fixture
def auth_app(state):
    """A minimal app exercising the verify_auth dependency via real HTTP."""
    app = FastAPI()
    app.state.auth = state

    @app.get("/protected", dependencies=[Depends(verify_auth())])
    def protected():
        return {"ok": True}

    @app.get("/scoped", dependencies=[Depends(verify_auth(["speech:generate"]))])
    def scoped():
        return {"ok": True}

    return app


def test_load_authorized_keys(state, key_file):
    load_authorized_keys(state, str(key_file))
    assert "test-key" in state.keys
    assert "speech:generate" in state.keys["test-key"]["scopes"]


def test_verify_jwt_valid(state, key_file):
    load_authorized_keys(state, str(key_file))
    token = _make_token(scopes=["speech:generate"])
    payload = _verify_jwt(f"Bearer {token}", state.keys, ["speech:generate"])
    assert payload["sub"] == "test-user"


def test_verify_jwt_missing_scope(state, key_file):
    load_authorized_keys(state, str(key_file))
    token = _make_token(scopes=["voices:read"])
    with pytest.raises(HTTPException) as exc_info:
        _verify_jwt(f"Bearer {token}", state.keys, ["speech:generate"])
    assert exc_info.value.status_code == 403
    assert "Missing required scope" in exc_info.value.detail


def test_verify_jwt_expired(state, key_file):
    load_authorized_keys(state, str(key_file))
    token = _make_token(expired=True)
    with pytest.raises(HTTPException) as exc_info:
        _verify_jwt(f"Bearer {token}", state.keys, [])
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


def test_verify_jwt_wrong_signature(state, key_file):
    load_authorized_keys(state, str(key_file))
    token = _make_token(wrong_key=True)
    with pytest.raises(HTTPException) as exc_info:
        _verify_jwt(f"Bearer {token}", state.keys, [])
    assert exc_info.value.status_code == 401


def test_verify_jwt_unknown_kid(state, key_file):
    load_authorized_keys(state, str(key_file))
    token = _make_token(kid="unknown-key")
    with pytest.raises(HTTPException) as exc_info:
        _verify_jwt(f"Bearer {token}", state.keys, [])
    assert exc_info.value.status_code == 401


def test_load_authorized_keys_openssh_ed25519(state, tmp_path):
    """OpenSSH-format public keys (ssh-keygen output) work with EdDSA JWTs."""
    private_key = ed25519.Ed25519PrivateKey.generate()
    ssh_pub_line = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    ).decode()
    path = tmp_path / "authorized_keys.json"
    path.write_text(
        json.dumps(
            {
                "keys": [
                    {
                        "kid": "ssh-key",
                        # Trailing comment, as found in real .pub files.
                        "public_key": f"{ssh_pub_line} user@host",
                        "scopes": ["speech:generate"],
                    }
                ]
            }
        )
    )
    load_authorized_keys(state, str(path))
    assert "ssh-key" in state.keys

    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "sub": "test-user",
            "iat": now,
            "exp": now + timedelta(hours=1),
            "scopes": ["speech:generate"],
        },
        private_key,
        algorithm="EdDSA",
        headers={"kid": "ssh-key"},
    )
    payload = _verify_jwt(f"Bearer {token}", state.keys, ["speech:generate"])
    assert payload["sub"] == "test-user"


def test_load_authorized_keys_malformed_file_fails_closed(state, tmp_path):
    """A malformed keys file must raise (fail closed), not leave the server public."""
    path = tmp_path / "authorized_keys.json"
    path.write_text("{ this is not json")
    with pytest.raises(RuntimeError, match="could not be parsed"):
        load_authorized_keys(state, str(path))
    assert not state.keys


def test_load_authorized_keys_skips_bad_entries(state, tmp_path):
    """Entries with unparseable keys are skipped, not fatal."""
    path = tmp_path / "authorized_keys.json"
    path.write_text(
        json.dumps(
            {
                "keys": [
                    {"kid": "bad-key", "public_key": "not-a-key", "scopes": []},
                    {
                        "kid": "good-key",
                        "public_key": PUBLIC_KEY,
                        "scopes": ["speech:generate"],
                    },
                ]
            }
        )
    )
    load_authorized_keys(state, str(path))
    assert "bad-key" not in state.keys
    assert "good-key" in state.keys


def test_dependency_public_when_no_keys(auth_app):
    """With no keys configured, endpoints are public."""
    resp = TestClient(auth_app).get("/protected")
    assert resp.status_code == 200


def test_dependency_missing_token_401(auth_app, state, key_file):
    load_authorized_keys(state, str(key_file))
    resp = TestClient(auth_app).get("/protected")
    assert resp.status_code == 401


def test_dependency_valid_token(auth_app, state, key_file):
    load_authorized_keys(state, str(key_file))
    token = _make_token(scopes=["speech:generate"])
    resp = TestClient(auth_app).get(
        "/scoped", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200


def test_dependency_missing_scope_403(auth_app, state, key_file):
    load_authorized_keys(state, str(key_file))
    token = _make_token(scopes=["voices:read"])
    resp = TestClient(auth_app).get(
        "/scoped", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 403
