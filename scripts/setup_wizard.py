#!/usr/bin/env python3
"""Interactive setup wizard for GLM-TTS-Server.

Personalizes a fresh clone: enrolls YOUR public key, adds YOUR voice(s)
(the first becomes the default), and commits the changes so CI builds a
personalized image. Standard library only - run with any Python 3.9+:

    python scripts/setup_wizard.py

The bundled voice and key are what make a fresh deploy "just work"; this
wizard replaces them with yours. Nothing is pushed automatically.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUTH_KEYS_PATH = REPO_ROOT / "authorized_keys.json"
VOICES_DIR = REPO_ROOT / "voices"
DOCKERFILE = REPO_ROOT / "Dockerfile"

DEFAULT_VOICE_PREFIX = "ENV GLM_TTS_DEFAULT_VOICE="
KEY_TYPE_PREFIXES = ("ssh-", "ecdsa-", "sk-")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_setup_wizard.py)
# ---------------------------------------------------------------------------

def validate_openssh_line(line: str) -> tuple[str, str]:
    """Validate an OpenSSH public-key line. Returns (normalized_line, comment)."""
    parts = line.split()
    if len(parts) < 2:
        raise ValueError("expected '<type> <base64> [comment]'")
    if not parts[0].startswith(KEY_TYPE_PREFIXES):
        raise ValueError(f"unrecognized key type: {parts[0]!r}")
    try:
        base64.b64decode(parts[1], validate=True)
    except Exception:
        raise ValueError("key material is not valid base64")
    return f"{parts[0]} {parts[1]}", " ".join(parts[2:])


def valid_voice_id(voice_id: str) -> bool:
    """Same rule as the server's upload endpoint: alphanumeric plus '-'/'_'."""
    return bool(voice_id) and all(c.isalnum() or c in "-_" for c in voice_id)


def with_comment(line: str, comment: str) -> str:
    """Reattach an OpenSSH comment (e.g. user@host) to a normalized key line."""
    return f"{line} {comment}" if comment else line


def read_dockerfile_default_voice(dockerfile: Path = DOCKERFILE) -> str:
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        if line.startswith(DEFAULT_VOICE_PREFIX):
            return line[len(DEFAULT_VOICE_PREFIX):].strip().strip('"')
    return ""


def set_dockerfile_default_voice(voice_id: str, dockerfile: Path = DOCKERFILE) -> str:
    """Rewrite (or append) the marked ENV line. Returns the previous value."""
    lines = dockerfile.read_text(encoding="utf-8").splitlines()
    old = ""
    for i, line in enumerate(lines):
        if line.startswith(DEFAULT_VOICE_PREFIX):
            old = line[len(DEFAULT_VOICE_PREFIX):].strip().strip('"')
            lines[i] = f"{DEFAULT_VOICE_PREFIX}{voice_id}"
            break
    else:
        lines += [
            "",
            "# Default voice when requests omit one. Personalized by scripts/setup_wizard.py.",
            f"{DEFAULT_VOICE_PREFIX}{voice_id}",
        ]
    dockerfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return old


def load_enrolled_keys(path: Path = AUTH_KEYS_PATH) -> list[dict]:
    """Read the keys file ({"keys": [{public_key, role}, ...]})."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    return [k for k in data.get("keys", []) if isinstance(k, dict)]


def write_authorized_keys(keys: list[dict], path: Path = AUTH_KEYS_PATH) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"keys": keys}, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or default


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    answer = input(f"{prompt} [{hint}]: ").strip().lower()
    if not answer:
        return default
    return answer.startswith("y")


def ask_menu(prompt: str, options: list[str]) -> int:
    """Print a numbered menu, return the 0-based choice index."""
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        raw = input(f"Choice [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print("  please enter a number from the list")


def detect_public_keys() -> list[tuple[Path, str, str]]:
    """Find ~/.ssh/*.pub files. Returns [(path, normalized_line, comment)]."""
    found = []
    for pub in sorted(Path.home().joinpath(".ssh").glob("*.pub")):
        try:
            line, comment = validate_openssh_line(pub.read_text(encoding="utf-8").strip())
            found.append((pub, line, comment))
        except (ValueError, OSError, UnicodeDecodeError):
            continue
    return found


# ---------------------------------------------------------------------------
# Wizard steps
# ---------------------------------------------------------------------------

def show_current_config() -> None:
    print("Current configuration:")
    keys = load_enrolled_keys()
    if keys:
        print(f"  auth:          {len(keys)} key(s) enrolled")
        for k in keys:
            try:
                _, comment = validate_openssh_line(k.get("public_key", ""))
            except ValueError:
                comment = "(unparseable)"
            print(f"                 - {comment or '(no comment)'} (role={k.get('role', 'user')})")
    else:
        print("  auth:          no keys enrolled (server is PUBLIC)")
    voices = sorted(p.name for p in VOICES_DIR.iterdir() if p.is_dir()) if VOICES_DIR.is_dir() else []
    print(f"  voices:        {', '.join(voices) or '(none)'}")
    print(f"  default voice: {read_dockerfile_default_voice() or '(auto-resolution)'}")


def step_auth(changed: list[str]) -> None:
    print("\n--- Step 1: authentication ---")
    choice = ask_menu(
        "How should the server authenticate clients?",
        [
            "SSH public key JWT (recommended)",
            "Open server (no auth - anyone with the URL can use it)",
            "Keep current configuration",
        ],
    )
    if choice == 2:
        return
    if choice == 1:
        print("\nWARNING: with no enrolled keys, ALL endpoints are publicly accessible.")
        if not ask_yes_no("Really disable authentication?", default=False):
            print("Keeping current configuration.")
            return
        write_authorized_keys([])
        changed.append("authorized_keys.json")
        print("authorized_keys.json: now empty (public mode).")
        return

    entries: list[dict] = []
    while True:
        detected = detect_public_keys()
        options = [f"{p}  ({c or 'no comment'})" for p, _, c in detected]
        options += ["Enter a path or paste an OpenSSH public key line"]
        if shutil.which("ssh-keygen"):
            options += ["Generate a new ed25519 keypair (ssh-keygen)"]
        idx = ask_menu("\nWhich key should be enrolled?", options)

        if idx < len(detected):
            line = with_comment(*detected[idx][1:])
        elif idx == len(detected):
            raw = ask("Path to .pub file, or paste the key line")
            candidate = Path(os.path.expanduser(raw))
            if candidate.is_file():
                raw = candidate.read_text(encoding="utf-8").strip()
            try:
                line = with_comment(*validate_openssh_line(raw))
            except ValueError as e:
                print(f"  invalid key: {e}; skipping")
                if not ask_yes_no("Try another key?"):
                    break
                continue
        else:
            line = generate_keypair()
            if line is None:
                if not ask_yes_no("Key generation failed. Choose another option?"):
                    break
                continue

        role = ask("Role (admin can manage voices; user is read+speech)", "admin")
        if role not in ("admin", "user"):
            print("  unknown role; using 'admin'")
            role = "admin"
        entries.append({"public_key": line, "role": role})
        print(f"  enrolled ({role}).")

        if not ask_yes_no("Add another key?", default=False):
            break

    if entries:
        write_authorized_keys(entries)
        changed.append("authorized_keys.json")
        print(f"authorized_keys.json: wrote {len(entries)} key(s).")


def generate_keypair() -> str | None:
    default_path = Path.home() / ".ssh" / "id_ed25519"
    raw = ask("Path for the new private key", str(default_path))
    priv = Path(os.path.expanduser(raw))
    if priv.exists():
        print(f"  {priv} already exists; not overwriting.")
        return None
    comment = ask("Key comment", f"glm-tts@{os.environ.get('USERNAME') or os.environ.get('USER') or 'me'}")
    priv.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(priv), "-N", "", "-C", comment],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ssh-keygen failed: {result.stderr.strip()}")
        return None
    print(f"  generated {priv} (no passphrase - needed for non-interactive token minting)")
    try:
        pub_path = priv.with_suffix(priv.suffix + ".pub")
        line, comment = validate_openssh_line(pub_path.read_text(encoding="utf-8").strip())
        return with_comment(line, comment)
    except (ValueError, OSError) as e:
        print(f"  could not read generated public key: {e}")
        return None


def step_voices(changed: list[str]) -> None:
    print("\n--- Step 2: your voice(s) ---")
    print("Each voice needs a short reference clip (3-10s of clean speech) and")
    print("the exact transcript of that clip. The FIRST voice you add becomes")
    print("the default voice baked into the image.")
    have_ffmpeg = shutil.which("ffmpeg") is not None
    if not have_ffmpeg:
        print("(ffmpeg not found: only .wav reference clips can be bundled;")
        print(" other formats can be uploaded via POST /v1/voices after deployment.)")

    added: list[str] = []
    pre_existing = sorted(p.name for p in VOICES_DIR.iterdir() if p.is_dir()) if VOICES_DIR.is_dir() else []
    while True:
        if added and not ask_yes_no("\nAdd another voice?", default=False):
            break

        voice_id = ask("\nvoice_id (alphanumeric, '-' or '_')", "my_voice" if not added else "")
        if not valid_voice_id(voice_id):
            print("  invalid voice_id; skipping this voice")
            continue
        target_dir = VOICES_DIR / voice_id
        if target_dir.exists():
            print(f"  voices/{voice_id} already exists; pick another id")
            continue

        audio_raw = ask("Path to reference audio")
        if not audio_raw:
            print("  no audio given; skipping this voice")
            if not added and not ask_yes_no("Skip voice setup entirely?", default=False):
                continue
            break
        audio_src = Path(os.path.expanduser(audio_raw))
        if not audio_src.is_file():
            print(f"  not found: {audio_src}")
            continue

        name = ask("Display name", voice_id)
        prompt_text = ask("Exact transcript of the reference clip")
        if not prompt_text:
            print("  transcript is required; skipping this voice")
            continue

        target_dir.mkdir(parents=True)
        try:
            if audio_src.suffix.lower() == ".wav" or not have_ffmpeg:
                if audio_src.suffix.lower() != ".wav":
                    print("  non-wav without ffmpeg - bundling as-is; the server may reject it")
                shutil.copyfile(audio_src, target_dir / "prompt_audio.wav")
            else:
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_src),
                     "-ar", "24000", "-ac", "1", "-sample_fmt", "s16",
                     str(target_dir / "prompt_audio.wav")],
                    check=True, capture_output=True, text=True,
                )
            (target_dir / "metadata.json").write_text(
                json.dumps({
                    "voice_id": voice_id,
                    "name": name,
                    "prompt_text": prompt_text,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as e:
            shutil.rmtree(target_dir, ignore_errors=True)
            print(f"  failed to add voice: {e}")
            continue

        added.append(voice_id)
        print(f"  added voices/{voice_id}/")
        if len(added) == 1:
            old = set_dockerfile_default_voice(voice_id)
            if "Dockerfile" not in changed:
                changed.append("Dockerfile")
            print(f"  default voice: {old or '(auto)'} -> {voice_id} (Dockerfile)")
        if "voices" not in changed:
            changed.append("voices")

    if not added:
        current = read_dockerfile_default_voice()
        print(f"No voices added; default voice stays {current or 'auto-resolution'}.")
        return

    # Personalization means YOUR voices - offer to remove the bundled ones.
    for voice_id in pre_existing:
        if voice_id in added:
            continue
        if ask_yes_no(f"Remove existing voice '{voice_id}'?", default=False):
            shutil.rmtree(VOICES_DIR / voice_id, ignore_errors=True)
            print(f"  removed voices/{voice_id}/")


def step_commit(changed: list[str]) -> None:
    print("\n--- Step 3: commit ---")
    if not changed:
        print("No changes to commit.")
        return
    print("Changed: " + ", ".join(changed))
    if not ask_yes_no("Commit these changes now?"):
        print("Left uncommitted; review with 'git status' and commit when ready.")
        return

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)

    if git("rev-parse", "--is-inside-work-tree").returncode != 0:
        print("Not a git repository; skipping commit.")
        return

    for field, prompt in (("user.name", "git user.name"), ("user.email", "git user.email")):
        if not git("config", field).stdout.strip():
            value = ask(f"{prompt} (repo-local, needed to commit)")
            if not value:
                print("  cannot commit without a git identity; skipping commit.")
                return
            git("config", field, value)

    git("add", "--", *changed)
    committed = git(
        "commit", "-m", "chore: personalize deployment via setup wizard", "--", *changed
    )
    if committed.returncode != 0:
        print(f"git commit failed:\n{committed.stderr.strip()}")
        return
    print("Committed.")
    print("Run 'git push' to trigger your CI build - the new image will carry")
    print("your key, your voices, and your default voice.")


def main() -> None:
    print("=" * 60)
    print("GLM-TTS-Server setup wizard")
    print("=" * 60)
    show_current_config()
    if not ask_yes_no("\nPersonalize this deployment?", default=False):
        print("\nNothing changed. To configure manually, see api/README.md")
        print("(authorized_keys.json, voices/, GLM_TTS_DEFAULT_VOICE).")
        return

    changed: list[str] = []
    try:
        step_auth(changed)
        step_voices(changed)
        step_commit(changed)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Files written so far: " + (", ".join(changed) or "none"))
        print("Commit manually, or re-run the wizard anytime.")
        sys.exit(130)


if __name__ == "__main__":
    main()
