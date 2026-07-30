#!/usr/bin/env python3
"""Mint a short-lived JWT for the GLM-TTS API.

Loads an OpenSSH or PEM private key (e.g. one generated with
``ssh-keygen -t ed25519 -f glm-tts-key``) and prints a signed JWT whose
``kid`` matches an entry in the server's authorized_keys.json.

Example:
    python scripts/make_token.py --key ~/.ssh/glm-tts-key --kid my-key
    TOKEN=$(python scripts/make_token.py --key ~/.ssh/glm-tts-key --kid my-key)
    curl -H "Authorization: Bearer $TOKEN" https://<pod>/v1/models
"""

from __future__ import annotations

import argparse
import getpass
from datetime import datetime, timedelta, timezone

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa

DEFAULT_SCOPES = ["speech:generate", "voices:read", "voices:manage"]


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
    for _attempt in range(3):
        try:
            return loader(data, password=password)
        except TypeError:
            # Encrypted key but no passphrase given yet.
            password = getpass.getpass(f"Passphrase for {path}: ").encode() or None
        except ValueError:
            if password is not None:
                # Most likely a wrong passphrase; retry.
                password = None
                continue
            raise SystemExit(f"error: cannot parse private key from {path}")
    raise SystemExit(f"error: cannot decrypt {path} (too many attempts)")


def _algorithm_for(key) -> str:
    if isinstance(key, ed25519.Ed25519PrivateKey):
        return "EdDSA"
    if isinstance(key, rsa.RSAPrivateKey):
        return "RS256"
    if isinstance(key, ec.EllipticCurvePrivateKey):
        return "ES256"
    raise SystemExit(f"error: unsupported key type: {type(key).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mint a short-lived JWT for the GLM-TTS API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--key", required=True, help="Path to the private key (OpenSSH or PEM).")
    parser.add_argument("--kid", required=True, help="Key ID matching an authorized_keys.json entry.")
    parser.add_argument("--sub", default="glm-tts-user", help="Subject claim (default: %(default)s).")
    parser.add_argument(
        "--scopes",
        nargs="+",
        default=DEFAULT_SCOPES,
        metavar="SCOPE",
        help="Scopes to grant (default: %(default)s).",
    )
    parser.add_argument(
        "--expires",
        type=int,
        default=3600,
        metavar="SECONDS",
        help="Token lifetime in seconds (default: %(default)s).",
    )
    args = parser.parse_args()

    private_key = _load_private_key(args.key)
    algorithm = _algorithm_for(private_key)

    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "sub": args.sub,
            "iat": now,
            "exp": now + timedelta(seconds=args.expires),
            "scopes": args.scopes,
        },
        private_key,
        algorithm=algorithm,
        headers={"kid": args.kid},
    )
    print(token)


if __name__ == "__main__":
    main()
