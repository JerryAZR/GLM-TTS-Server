# Unit tests for the setup wizard's pure helpers.
# Standard library only - runs locally without torch (like test_auth_unit.py).

import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from setup_wizard import (
    DEFAULT_VOICE_PREFIX,
    load_enrolled_keys,
    read_dockerfile_default_voice,
    set_dockerfile_default_voice,
    valid_voice_id,
    validate_openssh_line,
    with_comment,
    write_authorized_keys,
)

FAKE_B64 = base64.b64encode(b"fake-key-material").decode()


class TestValidateOpenSshLine:
    def test_valid_with_comment(self):
        line, comment = validate_openssh_line(f"ssh-ed25519 {FAKE_B64} me@host")
        assert line == f"ssh-ed25519 {FAKE_B64}"
        assert comment == "me@host"

    def test_valid_without_comment(self):
        line, comment = validate_openssh_line(f"ssh-rsa {FAKE_B64}")
        assert line == f"ssh-rsa {FAKE_B64}"
        assert comment == ""

    def test_normalizes_extra_whitespace(self):
        line, _ = validate_openssh_line(f"  ssh-ed25519   {FAKE_B64}   c  ")
        assert line == f"ssh-ed25519 {FAKE_B64}"

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            validate_openssh_line("not-a-key")
        with pytest.raises(ValueError):
            validate_openssh_line(f"ssh-ed25519 !!!not-base64!!!")
        with pytest.raises(ValueError):
            validate_openssh_line(f"unknown-type {FAKE_B64}")


class TestWithComment:
    def test_reattaches_comment(self):
        assert with_comment("ssh-ed25519 AAAA", "me@host") == "ssh-ed25519 AAAA me@host"

    def test_no_comment(self):
        assert with_comment("ssh-ed25519 AAAA", "") == "ssh-ed25519 AAAA"


class TestValidVoiceId:
    def test_accepts(self):
        assert valid_voice_id("jerry")
        assert valid_voice_id("my-voice_2")

    def test_rejects(self):
        assert not valid_voice_id("")
        assert not valid_voice_id("../etc")
        assert not valid_voice_id("has space")
        assert not valid_voice_id("slash/here")


class TestDockerfileDefaultVoice:
    def _write(self, path, lines):
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_rewrites_existing_line(self, tmp_path):
        df = tmp_path / "Dockerfile"
        self._write(df, ["FROM python:3.10", f"{DEFAULT_VOICE_PREFIX}jerry", "EXPOSE 8000"])
        old = set_dockerfile_default_voice("amy", df)
        assert old == "jerry"
        content = df.read_text(encoding="utf-8")
        assert f"{DEFAULT_VOICE_PREFIX}amy" in content
        assert "EXPOSE 8000" in content  # rest of file preserved
        assert content.count(DEFAULT_VOICE_PREFIX) == 1

    def test_appends_when_missing(self, tmp_path):
        df = tmp_path / "Dockerfile"
        self._write(df, ["FROM python:3.10"])
        old = set_dockerfile_default_voice("amy", df)
        assert old == ""
        assert read_dockerfile_default_voice(df) == "amy"

    def test_read_missing_returns_empty(self, tmp_path):
        df = tmp_path / "Dockerfile"
        self._write(df, ["FROM python:3.10"])
        assert read_dockerfile_default_voice(df) == ""


class TestAuthorizedKeysIo:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "authorized_keys.json"
        keys = [{"public_key": f"ssh-ed25519 {FAKE_B64} me@host", "role": "admin"}]
        write_authorized_keys(keys, path)
        assert load_enrolled_keys(path) == keys
        # canonical wrapper format the server expects
        assert json.loads(path.read_text(encoding="utf-8")) == {"keys": keys}

    def test_load_missing_or_invalid(self, tmp_path):
        assert load_enrolled_keys(tmp_path / "nope.json") == []
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert load_enrolled_keys(bad) == []
