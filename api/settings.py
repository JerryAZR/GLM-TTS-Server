"""
Runtime configuration for the GLM-TTS API server.

Settings are read from GLM_TTS_* environment variables when a Settings
object is constructed — never at module import time. Production creates one
Settings from the environment (see the module-level `app` in api.server);
tests construct their own explicitly, which keeps the test suite free of
env mutation and import-order hazards.
"""

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GLM_TTS_")

    auth_keys_file: str = ""
    model_dir: str = ""
    voices_dir: str = ""
    device: str = "auto"
    dtype: str = "float16"
    port: int = 8000
    mock_inference: bool = False
    use_phoneme: bool = False
    sample_rate: int = 24000
    max_upload_bytes: int = 20 * 1024 * 1024
    default_voice: str = ""
    git_sha: str = "unknown"

    @model_validator(mode="after")
    def _resolve_path_fallbacks(self) -> "Settings":
        """Default unset paths to the RunPod volume layout when present."""
        if not self.auth_keys_file:
            self.auth_keys_file = (
                "/workspace/authorized_keys.json"
                if Path("/workspace/authorized_keys.json").exists()
                else "authorized_keys.json"
            )
        if not self.model_dir:
            self.model_dir = (
                "/workspace/ckpt" if Path("/workspace/ckpt").exists() else "ckpt"
            )
        if not self.voices_dir:
            self.voices_dir = (
                "/workspace/voices" if Path("/workspace/voices").exists() else "voices"
            )
        return self
