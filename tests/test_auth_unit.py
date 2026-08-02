# Unit tests for api/auth.py.
# These tests do not import the heavy TTS model stack; they exercise the JWT
# public-key verification logic (try-all keys, role checks) against an
# AuthState and a minimal FastAPI app.

import json
from datetime import datetime, timezone, timedelta

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.auth import AuthState, load_authorized_keys, _verify_jwt, verify_auth


def _rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    return public_pem, private_pem


PUBLIC_KEY, PRIVATE_KEY = _rsa_keypair()
PUBLIC_KEY_2, PRIVATE_KEY_2 = _rsa_keypair()


def _make_token(private_key=PRIVATE_KEY, expired=False):
    now = datetime.now(timezone.utc)
    exp = now - timedelta(hours=1) if expired else now + timedelta(hours=1)
    return jwt.encode(
        {"sub": "test-user", "iat": now, "exp": exp},
        private_key,
        algorithm="RS256",
    )


def _keys_file(path, entries):
    path.write_text(json.dumps({"keys": entries}))
    return path


@pytest.fixture
def state():
    """A fresh AuthState per test — no global state to reset."""
    return AuthState()


@pytest.fixture
def key_file(tmp_path):
    return _keys_file(
        tmp_path / "authorized_keys.json",
        [{"public_key": PUBLIC_KEY, "role": "admin"}],
    )


@pytest.fixture
def auth_app(state):
    """A minimal app exercising the verify_auth dependency via real HTTP."""
    app = FastAPI()
    app.state.auth = state

    @app.get("/protected", dependencies=[Depends(verify_auth())])
    def protected():
        return {"ok": True}

    @app.get("/admin", dependencies=[Depends(verify_auth("admin"))])
    def admin():
        return {"ok": True}

    return app


def test_load_authorized_keys(state, key_file):
    load_authorized_keys(state, str(key_file))
    assert len(state.keys) == 1
    assert state.keys[0]["role"] == "admin"
    assert state.keys[0]["label"].startswith("sha256:")  # PEM has no comment


def test_verify_jwt_valid(state, key_file):
    load_authorized_keys(state, str(key_file))
    payload = _verify_jwt(f"Bearer {_make_token()}", state.keys, "user")
    assert payload["sub"] == "test-user"


def test_verify_jwt_try_all_keys(state, tmp_path):
    """A token without a key ID is verified against any enrolled key."""
    path = _keys_file(
        tmp_path / "authorized_keys.json",
        [
            {"public_key": PUBLIC_KEY, "role": "user"},
            {"public_key": PUBLIC_KEY_2, "role": "admin"},
        ],
    )
    load_authorized_keys(state, str(path))
    # Signed by the SECOND key: try-all must find it.
    payload = _verify_jwt(f"Bearer {_make_token(PRIVATE_KEY_2)}", state.keys, "admin")
    assert payload["sub"] == "test-user"


def test_verify_jwt_signature_matches_no_key(state, key_file):
    load_authorized_keys(state, str(key_file))
    with pytest.raises(HTTPException) as exc_info:
        _verify_jwt(f"Bearer {_make_token(PRIVATE_KEY_2)}", state.keys, "user")
    assert exc_info.value.status_code == 401
    assert "signature" in exc_info.value.detail.lower()


def test_verify_jwt_expired(state, key_file):
    load_authorized_keys(state, str(key_file))
    with pytest.raises(HTTPException) as exc_info:
        _verify_jwt(f"Bearer {_make_token(expired=True)}", state.keys, "user")
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


def test_verify_jwt_role_too_low(state, tmp_path):
    path = _keys_file(
        tmp_path / "authorized_keys.json",
        [{"public_key": PUBLIC_KEY, "role": "user"}],
    )
    load_authorized_keys(state, str(path))
    with pytest.raises(HTTPException) as exc_info:
        _verify_jwt(f"Bearer {_make_token()}", state.keys, "admin")
    assert exc_info.value.status_code == 403
    assert "admin" in exc_info.value.detail


def test_load_authorized_keys_openssh_label_from_comment(state, tmp_path):
    """OpenSSH keys work, and the .pub comment becomes the log label."""
    private_key = ed25519.Ed25519PrivateKey.generate()
    ssh_pub_line = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    ).decode()
    path = _keys_file(
        tmp_path / "authorized_keys.json",
        [{"public_key": f"{ssh_pub_line} user@host", "role": "admin"}],
    )
    load_authorized_keys(state, str(path))
    assert len(state.keys) == 1
    assert state.keys[0]["label"] == "user@host"

    token = jwt.encode(
        {
            "sub": "test-user",
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        },
        private_key,
        algorithm="EdDSA",
    )
    payload = _verify_jwt(f"Bearer {token}", state.keys, "admin")
    assert payload["sub"] == "test-user"


def test_load_authorized_keys_malformed_file_fails_closed(state, tmp_path):
    """A malformed keys file must raise (fail closed), not leave the server public."""
    path = tmp_path / "authorized_keys.json"
    path.write_text("{ this is not json")
    with pytest.raises(RuntimeError, match="could not be parsed"):
        load_authorized_keys(state, str(path))
    assert not state.keys


def test_load_authorized_keys_skips_bad_entries(state, tmp_path):
    """Entries with unparseable keys or unknown roles are skipped, not fatal."""
    path = _keys_file(
        tmp_path / "authorized_keys.json",
        [
            {"public_key": "not-a-key", "role": "admin"},
            {"public_key": PUBLIC_KEY_2, "role": "superuser"},
            {"public_key": PUBLIC_KEY, "role": "user"},
        ],
    )
    load_authorized_keys(state, str(path))
    assert len(state.keys) == 1
    assert state.keys[0]["role"] == "user"


def test_load_authorized_keys_default_role_is_user(state, tmp_path):
    path = _keys_file(
        tmp_path / "authorized_keys.json",
        [{"public_key": PUBLIC_KEY}],
    )
    load_authorized_keys(state, str(path))
    assert state.keys[0]["role"] == "user"


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
    resp = TestClient(auth_app).get(
        "/protected", headers={"Authorization": f"Bearer {_make_token()}"}
    )
    assert resp.status_code == 200


def test_dependency_admin_endpoint(auth_app, state, tmp_path):
    """A user-role token gets 403 on admin endpoints; admin gets 200."""
    path = _keys_file(
        tmp_path / "authorized_keys.json",
        [
            {"public_key": PUBLIC_KEY, "role": "user"},
            {"public_key": PUBLIC_KEY_2, "role": "admin"},
        ],
    )
    load_authorized_keys(state, str(path))
    client = TestClient(auth_app)
    resp = client.get("/admin", headers={"Authorization": f"Bearer {_make_token()}"})
    assert resp.status_code == 403
    resp = client.get("/admin", headers={"Authorization": f"Bearer {_make_token(PRIVATE_KEY_2)}"})
    assert resp.status_code == 200
