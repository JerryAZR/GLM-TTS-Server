# Unit tests for api/auth.py.
# These tests do not import the heavy TTS model stack; they only exercise the
# JWT public-key verification logic.

import asyncio
import json
import os
from datetime import datetime, timezone, timedelta

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ed25519
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

os.environ["GLM_TTS_AUTH_KEYS_FILE"] = "/nonexistent/authorized_keys.json"

from api.auth import (
    AUTHORIZED_KEYS,
    load_authorized_keys,
    _verify_jwt,
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
    yield
    auth.AUTHORIZED_KEYS.clear()


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


def test_load_authorized_keys_openssh_ed25519(tmp_path):
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
    load_authorized_keys(str(path))
    assert "ssh-key" in AUTHORIZED_KEYS

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
    payload = _verify_jwt(f"Bearer {token}", ["speech:generate"])
    assert payload["sub"] == "test-user"


def test_load_authorized_keys_malformed_file_fails_closed(tmp_path):
    """A malformed keys file must raise (fail closed), not leave the server public."""
    path = tmp_path / "authorized_keys.json"
    path.write_text("{ this is not json")
    with pytest.raises(RuntimeError, match="could not be parsed"):
        load_authorized_keys(str(path))
    assert not AUTHORIZED_KEYS


def test_load_authorized_keys_skips_bad_entries(tmp_path):
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
    load_authorized_keys(str(path))
    assert "bad-key" not in AUTHORIZED_KEYS
    assert "good-key" in AUTHORIZED_KEYS


def test_verify_jwt_unknown_kid(key_file):
    load_authorized_keys(str(key_file))
    token = _make_token(kid="unknown-key")
    with pytest.raises(HTTPException) as exc_info:
        _verify_jwt(f"Bearer {token}", [])
    assert exc_info.value.status_code == 401


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
