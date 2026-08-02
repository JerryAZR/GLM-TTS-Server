#!/usr/bin/env python3
"""Mint a short-lived JWT for the GLM-TTS API.

Loads an OpenSSH or PEM private key (e.g. one generated with
``ssh-keygen -t ed25519 -f glm-tts-key``) and prints a signed token that
proves possession of the key. Your privileges come from the key's enrolled
role on the server, not from the token.

Example:
    python scripts/make_token.py --key ~/.ssh/id_ed25519
    TOKEN=$(python scripts/make_token.py --key ~/.ssh/id_ed25519)
    curl -H "Authorization: Bearer $TOKEN" https://<pod>/v1/models
"""

from __future__ import annotations

import argparse
import getpass
import os
from datetime import datetime, timedelta, timezone

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa

# Default private key locations, tried in order when --key is omitted.
DEFAULT_KEY_PATHS = ("~/.ssh/id_ed25519", "~/.ssh/id_rsa")


def find_default_key() -> "str | None":
    """First existing default SSH private key, or None."""
    for candidate in DEFAULT_KEY_PATHS:
        path = os.path.expanduser(candidate)
        if os.path.isfile(path):
            return path
    return None


def _load_private_key(path: str):
    """Load a PEM or OpenSSH private key, prompting for a passphrase if needed."""
    with open(path, "rb") as f:
        data = f.read()

    header = data.lstrip()[:60]
    if header.startswith(b"-----BEGIN") and b"OPENSSH" not in header:
        loader = serialization.load_pem_private_key
    else:
        # OpenSSH private keys are armored too ("-----BEGIN OPENSSH PRIVATE
        # KEY-----"), so they must be detected before generic PEM.
        loader = serialization.load_ssh_private_key

    password = None
    prompts = 0
    while True:
        try:
            return loader(data, password=password)
        except TypeError:
            # Encrypted key but no passphrase given yet.
            pass
        except ValueError:
            if password is None:
                raise SystemExit(f"error: cannot parse private key from {path}")
            # Most likely a wrong passphrase; fall through to re-prompt.
        prompts += 1
        if prompts > 3:
            raise SystemExit(f"error: cannot decrypt {path} (too many attempts)")
        password = getpass.getpass(f"Passphrase for {path}: ").encode() or None


def _algorithm_for(key) -> str:
    if isinstance(key, ed25519.Ed25519PrivateKey):
        return "EdDSA"
    if isinstance(key, rsa.RSAPrivateKey):
        return "RS256"
    if isinstance(key, ec.EllipticCurvePrivateKey):
        return "ES256"
    raise SystemExit(f"error: unsupported key type: {type(key).__name__}")


def make_token(key_path: str, sub: str = "glm-tts-user", expires: int = 3600) -> str:
    """Sign a short-lived JWT with the private key at key_path."""
    private_key = _load_private_key(key_path)
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": sub,
            "iat": now,
            "exp": now + timedelta(seconds=expires),
        },
        private_key,
        algorithm=_algorithm_for(private_key),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mint a short-lived JWT for the GLM-TTS API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--key", default=None,
                        help="Path to the private key (OpenSSH or PEM). "
                             "Default: first of ~/.ssh/id_ed25519, ~/.ssh/id_rsa.")
    parser.add_argument("--sub", default="glm-tts-user", help="Subject claim (default: %(default)s).")
    parser.add_argument(
        "--expires",
        type=int,
        default=3600,
        metavar="SECONDS",
        help="Token lifetime in seconds (default: %(default)s).",
    )
    args = parser.parse_args()

    key_path = args.key or find_default_key()
    if not key_path:
        parser.error(
            "no --key given and no default key found at "
            + ", ".join(DEFAULT_KEY_PATHS)
        )
    print(make_token(key_path, sub=args.sub, expires=args.expires))


if __name__ == "__main__":
    main()
