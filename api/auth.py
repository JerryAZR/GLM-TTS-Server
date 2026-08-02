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

If no keys are configured, all /v1 endpoints are public. A keys file that
exists but cannot be parsed aborts startup (fail closed).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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

security = HTTPBearer(auto_error=False)


class AuthState:
    """Authorized public keys for one app instance (app.state.auth)."""

    def __init__(self) -> None:
        self.keys: Dict[str, Dict[str, Any]] = {}


def _load_public_key(key_text: str):
    """Load a PEM or OpenSSH-format (e.g. ssh-ed25519) public key."""
    data = key_text.strip().encode()
    if data.startswith((b"ssh-", b"ecdsa-")):
        return serialization.load_ssh_public_key(data)
    return serialization.load_pem_public_key(data)


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
        state.keys[kid] = {
            "public_key": key_obj,
            "scopes": set(entry.get("scopes", [])),
            "name": entry.get("name", ""),
        }

    logger.info(f"Loaded {len(state.keys)} authorized public key(s) from {path}")


def _verify_jwt(
    token: str, keys: Dict[str, Dict[str, Any]], required_scopes: List[str]
) -> Dict[str, Any]:
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
    if not kid or kid not in keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unknown or missing key ID",
        )

    key_entry = keys[kid]
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
    Reads the app's AuthState (app.state.auth): if public keys are
    configured, require a signed JWT; otherwise the endpoint is public.
    Returns the JWT payload (or an empty dict when no auth is configured).
    """
    required_scopes = required_scopes or []

    async def _verify(
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> Dict[str, Any]:
        keys = request.app.state.auth.keys

        # If no public keys are configured, the server is public.
        if not keys:
            return {}

        token = credentials.credentials if credentials else ""
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header",
            )

        if not token.startswith("Bearer "):
            token = f"Bearer {token}"
        return _verify_jwt(token, keys, required_scopes)

    return _verify
