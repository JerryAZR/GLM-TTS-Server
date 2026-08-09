"""
Authentication helpers for the GLM-TTS API server.

Public-key JWT auth: at startup the server loads a JSON file of authorized
public keys into an AuthState (stored on app.state.auth). Clients send
short-lived JWTs signed with their own private keys. The server only stores
public keys, never secrets.

Public keys may be PEM ("-----BEGIN PUBLIC KEY-----") or single-line OpenSSH
format ("ssh-ed25519 AAAA...", "ssh-rsa AAAA...", ...), so the file works
much like SSH's own authorized_keys: generate a key with ssh-keygen and
paste the .pub line.

Tokens carry no key ID: the server tries each enrolled key until the
signature verifies (signature verification is microseconds; enrolled key
counts are small). Each enrolled key has a role ("user" or "admin") that
governs which endpoints it may call; the token itself only proves
possession of the private key.

If no keys are configured, all /v1 endpoints are public. A keys file that
exists but cannot be parsed aborts startup (fail closed).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import jwt
from cryptography.hazmat.primitives import serialization
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# JWT algorithms we accept. Public-key JWTs should use one of these.
JWT_ALGORITHMS: List[str] = [
    "RS256",
    "RS384",
    "RS512",
    "ES256",
    "ES384",
    "ES512",
    "EdDSA",
]

# Role hierarchy: an entry may call endpoints requiring its level or lower.
ROLE_LEVELS = {"user": 1, "admin": 2}

security = HTTPBearer(auto_error=False)


class AuthState:
    """Authorized public keys for one app instance (app.state.auth)."""

    def __init__(self) -> None:
        self.keys: List[Dict[str, Any]] = []


def _load_public_key(key_text: str):
    """Load a PEM or OpenSSH-format (e.g. ssh-ed25519) public key."""
    data = key_text.strip().encode()
    if data.startswith((b"ssh-", b"ecdsa-")):
        return serialization.load_ssh_public_key(data)
    return serialization.load_pem_public_key(data)


def _key_label(key_text: str, key_obj) -> str:
    """Human-readable identity for logs: the OpenSSH comment if present
    (e.g. 'user@host' from ssh-keygen -C), else a short key fingerprint."""
    parts = key_text.strip().split()
    if parts and parts[0].startswith(("ssh-", "ecdsa-")) and len(parts) >= 3:
        return parts[2]
    digest = hashlib.sha256(
        key_obj.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    ).hexdigest()
    return f"sha256:{digest[:12]}"


def load_authorized_keys(state: AuthState, path: str) -> None:
    """Load pre-enrolled public keys from a JSON file into state.

    Missing file: state is left empty (server is public).
    Existing but unparseable file: RuntimeError (fail closed — a corrupted
    keys file must never silently open the server).
    """
    key_path = Path(path)
    state.keys.clear()
    if not key_path.exists():
        return

    try:
        data = json.loads(key_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Authorized keys file {path} exists but could not be parsed: {exc}"
        ) from exc

    for entry in data.get("keys", []):
        public_key = entry.get("public_key")
        if not public_key:
            logger.warning(f"Skipping authorized-key entry missing 'public_key': {entry}")
            continue
        role = entry.get("role", "user")
        if role not in ROLE_LEVELS:
            logger.warning(f"Skipping authorized-key entry with unknown role '{role}'")
            continue
        try:
            key_obj = _load_public_key(public_key)
        except Exception as exc:
            logger.warning(f"Skipping authorized-key entry (bad key): {exc}")
            continue
        state.keys.append(
            {
                "public_key": key_obj,
                "role": role,
                "label": _key_label(public_key, key_obj),
            }
        )

    logger.info(f"Loaded {len(state.keys)} authorized public key(s) from {path}")


def _verify_jwt(
    token: str, keys: List[Dict[str, Any]], required_role: str
) -> Dict[str, Any]:
    """Verify a JWT against the enrolled keys (try-all) and check its role."""
    if not token.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must use Bearer scheme",
        )
    jwt_token = token[7:]

    try:
        jwt.get_unverified_header(jwt_token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authorization token: {exc}",
        )

    payload = None
    matched = None
    for entry in keys:
        try:
            payload = jwt.decode(
                jwt_token,
                key=entry["public_key"],
                algorithms=JWT_ALGORITHMS,
                options={"require": ["exp", "sub"]},
            )
            matched = entry
            break
        except jwt.ExpiredSignatureError:
            # Signature verified but the token expired — no other key will help.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
            )
        except jwt.InvalidSignatureError:
            continue  # Not signed by this key; try the next one.
        except jwt.InvalidTokenError as exc:
            # Signature verified but claims are invalid — definitive.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {exc}",
            )

    if payload is None or matched is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token signature",
        )

    logger.debug(f"Authenticated key '{matched['label']}' (role={matched['role']})")
    if ROLE_LEVELS[matched["role"]] < ROLE_LEVELS[required_role]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"This endpoint requires the '{required_role}' role",
        )
    return payload


def verify_auth(required_role: str = "user"):
    """
    FastAPI dependency factory.
    Reads the app's AuthState (app.state.auth): if public keys are
    configured, require a JWT signed by an enrolled key whose role is at
    least required_role; otherwise the endpoint is public.
    """
    if required_role not in ROLE_LEVELS:
        raise ValueError(f"Unknown required_role: {required_role}")

    async def _verify(
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> Dict[str, Any]:
        keys = request.app.state.auth.keys

        # If no public keys are configured, the server is public.
        if not keys:
            return {}

        if credentials is None or not credentials.credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header",
            )

        return _verify_jwt(f"Bearer {credentials.credentials}", keys, required_role)

    return _verify
