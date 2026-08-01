"""
Authentication helpers for the GLM-TTS API server.

Public-key JWT auth: the server loads a JSON file of authorized public keys
(via GLM_TTS_AUTH_KEYS_FILE). Clients send short-lived JWTs signed with their
own private keys. The server only stores public keys, never secrets.

Public keys may be PEM ("-----BEGIN PUBLIC KEY-----") or single-line OpenSSH
format ("ssh-ed25519 AAAA...", "ssh-rsa AAAA...", ...), so the file works
much like SSH's own authorized_keys: generate a key with ssh-keygen and
paste the .pub line.

If no keys are configured, all /v1 endpoints are public.
"""

from __future__ import annotations

import json
import jwt
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

AUTH_KEYS_FILE = os.environ.get("GLM_TTS_AUTH_KEYS_FILE", "")
if not AUTH_KEYS_FILE:
    AUTH_KEYS_FILE = (
        "/workspace/authorized_keys.json"
        if Path("/workspace/authorized_keys.json").exists()
        else "authorized_keys.json"
    )

AUTHORIZED_KEYS: Dict[str, Dict[str, Any]] = {}

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

security = HTTPBearer(auto_error=False)


def _load_public_key(key_text: str):
    """Load a PEM or OpenSSH-format (e.g. ssh-ed25519) public key."""
    data = key_text.strip().encode()
    if data.startswith((b"ssh-", b"ecdsa-")):
        return serialization.load_ssh_public_key(data)
    return serialization.load_pem_public_key(data)


def load_authorized_keys(path: str) -> None:
    """Load pre-enrolled public keys from a JSON file."""
    key_path = Path(path)
    AUTHORIZED_KEYS.clear()
    if not key_path.exists():
        return

    try:
        with open(key_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        # Fail closed: a malformed keys file must not leave the server public.
        raise RuntimeError(
            f"Authorized keys file {path} exists but could not be parsed: {exc}"
        ) from exc

    for entry in data.get("keys", []):
        kid = entry.get("kid")
        public_key = entry.get("public_key")
        if not kid or not public_key:
            logger.warning(
                f"Skipping authorized-key entry missing 'kid' or 'public_key': {entry}"
            )
            continue
        try:
            key_obj = _load_public_key(public_key)
        except Exception as exc:
            logger.warning(f"Skipping authorized-key entry '{kid}' (bad key): {exc}")
            continue
        AUTHORIZED_KEYS[kid] = {
            "public_key": key_obj,
            "scopes": set(entry.get("scopes", [])),
            "name": entry.get("name", ""),
        }

    logger.info(f"Loaded {len(AUTHORIZED_KEYS)} authorized public key(s) from {path}")


def _verify_jwt(token: str, required_scopes: List[str]) -> Dict[str, Any]:
    """Verify a JWT signed with a pre-enrolled private key."""
    if not token.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must use Bearer scheme",
        )
    jwt_token = token[7:]

    try:
        header = jwt.get_unverified_header(jwt_token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authorization token: {exc}",
        )

    kid = header.get("kid")
    if not kid or kid not in AUTHORIZED_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unknown or missing key ID",
        )

    key_entry = AUTHORIZED_KEYS[kid]
    try:
        payload = jwt.decode(
            jwt_token,
            key=key_entry["public_key"],
            algorithms=JWT_ALGORITHMS,
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        )

    token_scopes: Set[str] = set(payload.get("scopes", []))
    for scope in required_scopes:
        if scope not in token_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scope: {scope}",
            )

    return payload


def verify_auth(required_scopes: Optional[List[str]] = None):
    """
    FastAPI dependency factory.
    If public keys are configured, require a signed JWT; otherwise the
    endpoint is public. Returns the JWT payload (or an empty dict when
    no auth is configured).
    """
    required_scopes = required_scopes or []

    async def _verify(
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> Dict[str, Any]:
        # If no public keys are configured, the server is public.
        if not AUTHORIZED_KEYS:
            return {}

        token = credentials.credentials if credentials else ""
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header",
            )

        if not token.startswith("Bearer "):
            token = f"Bearer {token}"
        return _verify_jwt(token, required_scopes)

    return _verify
