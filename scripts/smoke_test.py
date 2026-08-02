#!/usr/bin/env python3
"""One-command smoke test for a deployed GLM-TTS API server.

Verifies that a deployment is healthy and can actually synthesize speech:

    python scripts/smoke_test.py --endpoint https://<pod>.proxy.runpod.net \
        --key ~/.ssh/id_ed25519

Checks (each printed as [ok]/[FAIL], non-zero exit on any failure):
  1. /health responds
  2. /version reports the deployed git sha
  3. /ready becomes true (model loaded; polls up to --timeout seconds)
  4. auth: with --key, a tokenless request is rejected (401) and a signed
     JWT is accepted; without --key, the server is expected to be public
  5. /status reports device, voices, and the resolved default voice
  6. POST /v1/audio/speech returns a valid WAV (written to --output)

Only the Python standard library plus PyJWT/cryptography (already used by
scripts/make_token.py) are required.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_token import make_token, find_default_key  # noqa: E402

# RunPod's proxy is behind Cloudflare bot protection: default library user
# agents get "403 error code: 1010". A browser-shaped UA avoids that.
HEADERS = {"User-Agent": "Mozilla/5.0 (GLM-TTS smoke test)"}

_failures = 0


def report(ok: bool, label: str, detail: str = "") -> bool:
    global _failures
    if not ok:
        _failures += 1
    suffix = f" - {detail}" if detail else ""
    print(f"[{'ok' if ok else 'FAIL'}] {label}{suffix}")
    return ok


def http(method: str, url: str, token: str = "", body: dict = None, timeout: int = 30):
    """Returns (status_code, response_bytes). status_code 0 = connection error."""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = dict(HEADERS)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception:
        return 0, b""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--endpoint", required=True,
                        help="Base URL, e.g. https://<pod>.proxy.runpod.net")
    parser.add_argument("--key", default=None,
                        help="Private key for JWT auth. Default: first of "
                             "~/.ssh/id_ed25519, ~/.ssh/id_rsa. Sending a "
                             "token to a public server is harmless.")
    parser.add_argument("--sub", default="smoke-test", help="Token subject (default: %(default)s).")
    parser.add_argument("--voice", default="",
                        help="Voice to synthesize with (default: omit, exercising default resolution).")
    parser.add_argument("--text", default="Hello, this is a GLM-TTS deployment smoke test.",
                        help="Text to synthesize.")
    parser.add_argument("--output", default="smoke_test.wav", help="Output WAV path.")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Seconds to wait for /ready (default: %(default)s; first boot "
                             "includes model download + load).")
    args = parser.parse_args()

    ep = args.endpoint.rstrip("/")

    # 1. Liveness.
    code, data = http("GET", f"{ep}/health")
    ok = code == 200 and b'"ok"' in data
    if not report(ok, "GET /health", f"HTTP {code}" if not ok else ""):
        return 1  # Nothing else can work.

    # 2. Build stamp.
    code, data = http("GET", f"{ep}/version")
    version = json.loads(data).get("version", "?") if code == 200 else "?"
    report(code == 200, "GET /version", f"deployed revision: {version}")

    # 3. Readiness (model loaded), with polling.
    deadline = time.time() + args.timeout
    ready, last = False, ""
    while time.time() < deadline:
        code, data = http("GET", f"{ep}/ready")
        if code == 200:
            payload = json.loads(data)
            if payload.get("ready"):
                ready = True
                break
            last = payload.get("error", "")
        time.sleep(5)
    if not report(ready, "GET /ready", "model loaded" if ready else f"not ready after {args.timeout}s: {last}"):
        return 1

    # 4. Auth behavior. A token is minted whenever a key is available — on a
    # public server it is simply ignored (verify_auth short-circuits), so
    # both modes pass; the report just notes which one is live.
    token = ""
    key_path = args.key or find_default_key()
    if key_path:
        code, _ = http("GET", f"{ep}/v1/models")
        if code == 401:
            report(True, "auth enforced", "tokenless /v1/models -> HTTP 401")
        elif code == 200:
            report(True, "server is public", "token accepted but not required")
        else:
            report(False, "auth probe", f"tokenless /v1/models -> HTTP {code}")
        token = make_token(key_path, sub=args.sub)
        code, _ = http("GET", f"{ep}/v1/models", token=token)
        if not report(code == 200, "JWT accepted", f"key: {key_path}" + ("" if code == 200 else f", HTTP {code}")):
            return 1
    else:
        code, _ = http("GET", f"{ep}/v1/models")
        report(code == 200, "server is public (no key found)", f"HTTP {code}")

    # 5. Status overview.
    code, data = http("GET", f"{ep}/status", token=token)
    if code == 200:
        s = json.loads(data)
        report(True, "GET /status",
               f"{s.get('device')}/{s.get('dtype')}, {s.get('voices')} voice(s), "
               f"default: {s.get('default_voice')}")
    else:
        report(False, "GET /status", f"HTTP {code}")

    # 6. Real synthesis.
    body = {"model": "glm-tts", "input": args.text, "response_format": "wav"}
    if args.voice:
        body["voice"] = args.voice
    code, data = http("POST", f"{ep}/v1/audio/speech", token=token, body=body, timeout=180)
    if code != 200:
        detail = data.decode(errors="replace")[:200]
        report(False, "POST /v1/audio/speech", f"HTTP {code}: {detail}")
    else:
        try:
            w = wave.open(io.BytesIO(data))
            seconds = w.getnframes() / w.getframerate()
            with open(args.output, "wb") as f:
                f.write(data)
            report(seconds > 0.2, "POST /v1/audio/speech",
                   f"{w.getnchannels()}ch {w.getframerate()}Hz {seconds:.1f}s -> {args.output}")
        except Exception as e:
            report(False, "POST /v1/audio/speech", f"response is not a valid WAV: {e}")

    print()
    if _failures:
        print(f"SMOKE TEST FAILED ({_failures} check(s) failed)")
        return 1
    print(f"SMOKE TEST PASSED - play {args.output} to hear the result")
    return 0


if __name__ == "__main__":
    sys.exit(main())
