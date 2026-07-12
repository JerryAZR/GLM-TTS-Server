# Unit tests for api/auth.py.
# These tests do not import the heavy TTS model stack; they only exercise the
# JWT and legacy-API-key verification logic.

import asyncio
import json
import os
from datetime import datetime, timezone, timedelta

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

os.environ["GLM_TTS_API_KEY"] = ""
os.environ["GLM_TTS_AUTH_KEYS_FILE"] = "/nonexistent/authorized_keys.json"

from api.auth import (
    AUTHORIZED_KEYS,
    load_authorized_keys,
    _verify_jwt,
    _legacy_verify,
    verify_auth,
)


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


@pytest.fixture(autouse=True)
def reset_auth_keys():
    """Reset the in-memory authorized key set before each test."""
    import api.auth as auth

    auth.AUTHORIZED_KEYS.clear()
    auth.LEGACY_API_KEY = ""
    yield
    auth.AUTHORIZED_KEYS.clear()
    auth.LEGACY_API_KEY = ""


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


def test_load_authorized_keys(key_file):
    load_authorized_keys(str(key_file))
    assert "test-key" in AUTHORIZED_KEYS
    assert "speech:generate" in AUTHORIZED_KEYS["test-key"]["scopes"]


def test_verify_jwt_valid(key_file):
    load_authorized_keys(str(key_file))
    token = _make_token(scopes=["speech:generate"])
    payload = _verify_jwt(f"Bearer {token}", ["speech:generate"])
    assert payload["sub"] == "test-user"


def test_verify_jwt_missing_scope(key_file):
    load_authorized_keys(str(key_file))
    token = _make_token(scopes=["voices:read"])
    with pytest.raises(HTTPException) as exc_info:
        _verify_jwt(f"Bearer {token}", ["speech:generate"])
    assert exc_info.value.status_code == 403
    assert "Missing required scope" in exc_info.value.detail


def test_verify_jwt_expired(key_file):
    load_authorized_keys(str(key_file))
    token = _make_token(expired=True)
    with pytest.raises(HTTPException) as exc_info:
        _verify_jwt(f"Bearer {token}", [])
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


def test_verify_jwt_wrong_signature(key_file):
    load_authorized_keys(str(key_file))
    token = _make_token(wrong_key=True)
    with pytest.raises(HTTPException) as exc_info:
        _verify_jwt(f"Bearer {token}", [])
    assert exc_info.value.status_code == 401


def test_verify_jwt_unknown_kid(key_file):
    load_authorized_keys(str(key_file))
    token = _make_token(kid="unknown-key")
    with pytest.raises(HTTPException) as exc_info:
        _verify_jwt(f"Bearer {token}", [])
    assert exc_info.value.status_code == 401


def test_legacy_verify(monkeypatch):
    monkeypatch.setenv("GLM_TTS_API_KEY", "secret")
    # LEGACY_API_KEY is read at import time; reload to pick up the env var.
    import importlib
    import api.auth

    importlib.reload(api.auth)
    from api.auth import _legacy_verify

    assert _legacy_verify("secret") is True
    assert _legacy_verify("Bearer secret") is True
    assert _legacy_verify("wrong") is False


@pytest.mark.asyncio
async def test_verify_auth_dependency_public_when_no_auth():
    """When no auth is configured, the dependency returns an empty dict."""
    AUTHORIZED_KEYS.clear()
    dep = verify_auth()
    result = await dep(None)
    assert result == {}


@pytest.mark.asyncio
async def test_verify_auth_dependency_missing_token(key_file):
    load_authorized_keys(str(key_file))
    dep = verify_auth()
    with pytest.raises(HTTPException) as exc_info:
        await dep(None)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_auth_dependency_valid_token(key_file):
    load_authorized_keys(str(key_file))
    token = _make_token(scopes=["speech:generate"])
    dep = verify_auth(["speech:generate"])
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    result = await dep(creds)
    assert result["sub"] == "test-user"
