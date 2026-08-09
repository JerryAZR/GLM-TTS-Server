"""
OpenAI-compatible FastAPI server for GLM-TTS (zero-shot voice cloning TTS).

Endpoints:
  GET  /health           - liveness (container healthcheck)
  GET  /ready            - model load status
  GET  /version          - image build stamp
  GET  /status           - config, uptime, request stats (auth required)
  GET  /v1/models        - model list (auth required)
  POST /v1/audio/speech  - TTS generation (scope: speech:generate)
  /v1/voices             - voice CRUD (scopes: voices:read / voices:manage)

Configuration comes from api.settings.Settings (GLM_TTS_* env vars), read
once when the app is created. All mutable state (auth keys, voice registry,
inference engine, stats) lives on app.state, created per app instance in
create_app(). The module-level `app` at the bottom is the uvicorn entry
point; tests build their own instances with explicit Settings instead of
mutating the environment.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import random
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import soundfile as sf
import torch
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pydub import AudioSegment

from api.auth import AuthState, load_authorized_keys, verify_auth
from api.settings import Settings
from glmtts_inference import generate_long, load_models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}
_HALF_DTYPES = ("float16", "fp16", "half", "bfloat16", "bf16")


# ---------------------------------------------------------------------------
# Voice registry helpers
# ---------------------------------------------------------------------------

class VoiceEntry(BaseModel):
    voice_id: str
    name: str
    prompt_text: str
    created_at: str
    path: Path


def _pick_default(voices: Dict[str, VoiceEntry], configured_default: str = "") -> Optional[str]:
    """Best-effort default voice, never raises. Priority:
    configured default > voice_id "default" > sole registered voice."""
    if configured_default and configured_default in voices:
        return configured_default
    if "default" in voices:
        return "default"
    if len(voices) == 1:
        return next(iter(voices))
    return None


def resolve_voice(
    requested: Optional[str],
    voices: Dict[str, VoiceEntry],
    configured_default: str = "",
) -> str:
    """Resolve which voice a speech request uses. Called per request (the
    registry is mutable via the voices API; resolution is trivial).

    Priority: explicit request > configured default (GLM_TTS_DEFAULT_VOICE)
    > voice_id "default" > sole registered voice. Failures are loud and
    precise: a dangling configured default is an operator error, never a
    silent fallthrough to a different voice.
    """
    if requested:
        if requested not in voices:
            raise HTTPException(
                status_code=404,
                detail=f"Voice '{requested}' not found; available: {sorted(voices)}",
            )
        return requested
    if configured_default and configured_default not in voices:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Configured default voice '{configured_default}' "
                f"(GLM_TTS_DEFAULT_VOICE) not found; available: {sorted(voices)}"
            ),
        )
    default = _pick_default(voices, configured_default)
    if default is not None:
        return default
    if not voices:
        raise HTTPException(status_code=404, detail="No voices available")
    raise HTTPException(
        status_code=400,
        detail=f"Multiple voices available; specify 'voice': {sorted(voices)}",
    )


def _voice_meta_path(voice_path: Path) -> Path:
    return voice_path / "metadata.json"


def _voice_audio_path(voice_path: Path) -> Path:
    return voice_path / "prompt_audio.wav"


def _load_voice_from_disk(voice_path: Path) -> Optional[VoiceEntry]:
    meta_path = _voice_meta_path(voice_path)
    audio_path = _voice_audio_path(voice_path)
    if not meta_path.exists() or not audio_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read voice metadata {meta_path}: {e}")
        return None

    return VoiceEntry(
        voice_id=meta.get("voice_id", voice_path.name),
        name=meta.get("name", voice_path.name),
        prompt_text=meta.get("prompt_text", ""),
        created_at=meta.get("created_at", datetime.now(timezone.utc).isoformat()),
        path=voice_path,
    )


def scan_voices(voices_dir: str, voices: Dict[str, VoiceEntry]) -> None:
    """Scan the voices directory and populate the in-memory registry."""
    voices.clear()
    voices_root = Path(voices_dir)
    voices_root.mkdir(parents=True, exist_ok=True)
    for entry in voices_root.iterdir():
        if not entry.is_dir():
            continue
        voice = _load_voice_from_disk(entry)
        if voice:
            voices[voice.voice_id] = voice
    logger.info(f"Loaded {len(voices)} voice(s) from {voices_root}")


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """Holds the loaded models and serializes generation with a lock."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device: Optional[torch.device] = None
        self.dtype: Optional[torch.dtype] = None
        self.frontend = None
        self.text_frontend = None
        self.speech_tokenizer = None
        self.llm = None
        self.flow = None  # This is actually the Token2Wav wrapper
        self.lock = asyncio.Lock()
        self.ready = False
        self.startup_error: Optional[str] = None

    def configure(self) -> None:
        device_str = self.settings.device
        if device_str.lower() == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        dtype_str = self.settings.dtype.lower()
        if self.device.type == "cpu" and dtype_str in _HALF_DTYPES:
            logger.warning(f"Device is CPU; forcing float32 instead of {self.settings.dtype}")
            dtype_str = "float32"
        self.dtype = _DTYPES.get(dtype_str)
        if self.dtype is None:
            raise ValueError(f"Unknown dtype: {self.settings.dtype}")
        logger.info(f"Inference engine configured: device={self.device}, dtype={self.dtype}")

    def load(self) -> None:
        if self.settings.mock_inference:
            logger.info("Mock inference enabled; skipping model load.")
            self.ready = True
            return

        self.configure()
        logger.info(f"Loading models from {self.settings.model_dir} ...")
        (
            self.frontend,
            self.text_frontend,
            self.speech_tokenizer,
            self.llm,
            self.flow,
        ) = load_models(
            use_phoneme=self.settings.use_phoneme,
            sample_rate=self.settings.sample_rate,
            device=self.device,
            dtype=self.dtype,
            model_dir=self.settings.model_dir,
        )
        self.ready = True
        logger.info("Models loaded successfully.")

    def _extract_prompt_features(self, voice: VoiceEntry):
        audio_path = voice.path / "prompt_audio.wav"
        prompt_text = self.text_frontend.text_normalize(voice.prompt_text)
        prompt_text_token = self.frontend._extract_text_token(prompt_text + " ")
        prompt_speech_token = self.frontend._extract_speech_token([str(audio_path)])
        speech_feat = self.frontend._extract_speech_feat(
            str(audio_path), sample_rate=self.settings.sample_rate
        )
        embedding = self.frontend._extract_spk_embedding(str(audio_path))

        # Keep audio features in fp32; only the LLM is cast to the configured dtype.
        speech_feat = speech_feat.to(device=self.device)
        embedding = embedding.to(device=self.device)

        cache_speech_token = [prompt_speech_token.squeeze().tolist()]
        flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(self.device)

        return prompt_text, prompt_text_token, cache_speech_token, flow_prompt_token, speech_feat, embedding

    def synthesize(self, text: str, voice: VoiceEntry, seed: int = 0) -> torch.Tensor:
        if self.settings.mock_inference:
            # 1 second of silence-ish dummy sine wave at the target sample rate
            t = torch.linspace(0, 1, self.settings.sample_rate, dtype=torch.float32)
            return torch.sin(2 * 3.1415926 * 440 * t).unsqueeze(0)

        audio_path = voice.path / "prompt_audio.wav"
        if not audio_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Voice audio for '{voice.voice_id}' not found"
            )

        (
            prompt_text,
            prompt_text_token,
            cache_speech_token,
            flow_prompt_token,
            speech_feat,
            embedding,
        ) = self._extract_prompt_features(voice)

        cache = {
            "cache_text": [prompt_text],
            "cache_text_token": [prompt_text_token],
            "cache_speech_token": cache_speech_token,
            "use_cache": True,
        }

        tts_speech, _, _, _ = generate_long(
            frontend=self.frontend,
            text_frontend=self.text_frontend,
            llm=self.llm,
            flow=self.flow,
            text_info=[voice.voice_id, text],
            cache=cache,
            embedding=embedding,
            device=self.device,
            flow_prompt_token=flow_prompt_token,
            speech_feat=speech_feat,
            use_phoneme=self.settings.use_phoneme,
            seed=seed,
        )
        return tts_speech


# ---------------------------------------------------------------------------
# Degenerate-output detection and retry
# ---------------------------------------------------------------------------

# Normal speech is ~3-4 words/sec (English) or ~3-5 chars/sec (Chinese), so
# output shorter than this fraction of the text's plausible minimum indicates
# the LLM emitted EOS almost immediately (a deterministic-seed sampling
# failure). Gated at _MIN_UNITS so short utterances ("Hi.") never trigger.
_MIN_SECONDS_PER_UNIT = 0.15
_MIN_UNITS = 4
MAX_GENERATION_ATTEMPTS = 3


def _text_units(text: str) -> int:
    """Rough spoken-length units: CJK characters plus non-CJK words."""
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    non_cjk = "".join(" " if "\u4e00" <= c <= "\u9fff" else c for c in text)
    return cjk + len(non_cjk.split())


def _is_degenerate_output(text: str, seconds: float) -> bool:
    units = _text_units(text)
    return units >= _MIN_UNITS and seconds < _MIN_SECONDS_PER_UNIT * units


async def _generate_with_retry(
    engine: InferenceEngine,
    text: str,
    voice: VoiceEntry,
    sample_rate: int,
    max_attempts: int = MAX_GENERATION_ATTEMPTS,
):
    """Generate speech, retrying with fresh random seeds when the output is
    implausibly short for the text. Returns (wav, degenerate_flag); when all
    attempts are degenerate, returns the longest one with the flag set."""
    best_wav, best_seconds = None, -1.0
    units = _text_units(text)
    for attempt in range(1, max_attempts + 1):
        seed = random.randint(0, 2**31 - 1)
        wav = await asyncio.to_thread(engine.synthesize, text, voice, seed)
        seconds = wav.shape[-1] / sample_rate
        if seconds > best_seconds:
            best_wav, best_seconds = wav, seconds
        if not _is_degenerate_output(text, seconds):
            return best_wav, False
        logger.warning(
            f"Degenerate output ({seconds:.2f}s for {units}-unit text) with seed {seed}"
            + ("; retrying with a new seed" if attempt < max_attempts else "; returning best attempt")
        )
    return best_wav, True

def _tensor_to_wav_bytes(wav: torch.Tensor, sample_rate: int) -> bytes:
    wav = wav.detach().cpu().squeeze().numpy()
    if wav.ndim == 1:
        wav = wav[None, :]
    buf = io.BytesIO()
    sf.write(buf, wav.T, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def _wav_to_mp3_bytes(wav_bytes: bytes) -> bytes:
    try:
        audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        buf = io.BytesIO()
        audio.export(buf, format="mp3")
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error(f"MP3 conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"MP3 conversion failed: {e}")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    model: str = "glm-tts"
    input: str = Field(..., min_length=1, max_length=5000)
    voice: Optional[str] = None  # resolved via resolve_voice() when omitted
    response_format: str = "wav"  # wav or mp3
    speed: float = 1.0


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(settings: Settings) -> FastAPI:
    """Build the FastAPI app for the given settings.

    All mutable state (auth keys, voice registry, engine, stats) is created
    per instance and exposed on app.state, so apps are independent and
    tests never share state with production or each other.
    """
    auth = AuthState()
    voices: Dict[str, VoiceEntry] = {}
    voice_lock = asyncio.Lock()
    engine = InferenceEngine(settings)
    started_at = time.time()
    # Lightweight request stats (in-memory; reset on restart). A "quick
    # check" for /status — anything heavier belongs in an SSH session.
    stats = {
        "speech_requests": 0,
        "failed_requests": 0,
        "audio_seconds_generated": 0.0,
        "last_generation_seconds": None,
    }

    async def _load_engine() -> None:
        """Load models in a background thread so /health stays responsive."""
        try:
            await asyncio.to_thread(engine.load)
        except Exception as exc:
            logger.exception("Inference engine failed to load")
            engine.ready = False
            engine.startup_error = str(exc)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail-closed: a malformed keys file aborts startup (see auth.py).
        load_authorized_keys(auth, settings.auth_keys_file)
        scan_voices(settings.voices_dir, voices)
        if settings.default_voice and settings.default_voice not in voices:
            logger.warning(
                f"GLM_TTS_DEFAULT_VOICE='{settings.default_voice}' does not match any "
                f"registered voice; speech requests without an explicit voice will fail"
            )
        else:
            logger.info(f"Default voice: {_pick_default(voices, settings.default_voice)}")
        asyncio.create_task(_load_engine())
        yield

    app = FastAPI(title="GLM-TTS Server", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.auth = auth
    app.state.voices = voices
    app.state.engine = engine
    app.state.stats = stats
    app.state.started_at = started_at

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        payload = {"ready": engine.ready, "mock": settings.mock_inference}
        if not engine.ready and engine.startup_error:
            payload["error"] = engine.startup_error
        return payload

    @app.get("/version")
    def version():
        """Public build stamp: which image revision is running."""
        return {"version": settings.git_sha, "mock": settings.mock_inference}

    @app.get("/status")
    def server_status(_=Depends(verify_auth())):
        """Quick operational check: config, uptime, and request stats."""
        payload = {
            "version": settings.git_sha,
            "ready": engine.ready,
            "mock": settings.mock_inference,
            "device": str(engine.device) if engine.device is not None else settings.device,
            "dtype": (
                str(engine.dtype).replace("torch.", "")
                if engine.dtype is not None
                else settings.dtype
            ),
            "sample_rate": settings.sample_rate,
            "uptime_seconds": round(time.time() - started_at, 1),
            "voices": len(voices),
            "default_voice": _pick_default(voices, settings.default_voice),
            "generating": engine.lock.locked(),
            "stats": dict(stats),
        }
        if not engine.ready and engine.startup_error:
            payload["startup_error"] = engine.startup_error
        return payload

    @app.get("/v1/models")
    def list_models(_=Depends(verify_auth())):
        return {"data": [{"id": "glm-tts", "object": "model", "owned_by": "zai-org"}]}

    @app.post("/v1/audio/speech")
    async def create_speech(
        req: SpeechRequest, _=Depends(verify_auth())
    ):
        if not engine.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference engine is not ready yet",
            )

        if req.response_format not in ("wav", "mp3"):
            raise HTTPException(
                status_code=400, detail="response_format must be 'wav' or 'mp3'"
            )

        if not req.input.strip():
            raise HTTPException(status_code=400, detail="input text cannot be empty")

        voice_id = resolve_voice(req.voice, voices, settings.default_voice)
        voice = voices[voice_id]

        stats["speech_requests"] += 1
        async with engine.lock:
            t0 = time.monotonic()
            try:
                wav, degenerate = await _generate_with_retry(
                    engine, req.input, voice, settings.sample_rate
                )
            except Exception:
                stats["failed_requests"] += 1
                raise
            stats["last_generation_seconds"] = round(time.monotonic() - t0, 2)
            stats["audio_seconds_generated"] = round(
                stats["audio_seconds_generated"] + wav.shape[-1] / settings.sample_rate, 2
            )
            wav_bytes = _tensor_to_wav_bytes(wav, settings.sample_rate)

        if req.response_format == "mp3":
            audio_bytes = _wav_to_mp3_bytes(wav_bytes)
            media_type = "audio/mpeg"
        else:
            audio_bytes = wav_bytes
            media_type = "audio/wav"

        headers = {"X-GLM-TTS-Warning": "degenerate-output"} if degenerate else None
        return Response(content=audio_bytes, media_type=media_type, headers=headers)

    @app.post("/v1/voices")
    async def create_voice(
        name: str = Form(...),
        prompt_text: str = Form(...),
        prompt_audio: UploadFile = File(...),
        voice_id: Optional[str] = Form(None),
        _=Depends(verify_auth("admin")),
    ):
        if not voice_id:
            voice_id = f"voice_{uuid.uuid4().hex[:12]}"
        voice_id = voice_id.strip()
        if not voice_id:
            raise HTTPException(status_code=400, detail="voice_id cannot be empty")

        # Only allow simple IDs to avoid path traversal
        safe_id = "".join(c for c in voice_id if c.isalnum() or c in "-_")
        if not safe_id or safe_id != voice_id:
            raise HTTPException(
                status_code=400, detail="voice_id must be alphanumeric with '-' or '_'"
            )

        content = await prompt_audio.read()
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Audio file exceeds {settings.max_upload_bytes} bytes "
                    f"({len(content)} bytes uploaded)"
                ),
            )

        async with voice_lock:
            voice_path = Path(settings.voices_dir) / safe_id
            if voice_path.exists():
                raise HTTPException(status_code=409, detail=f"Voice '{safe_id}' already exists")

            voice_path.mkdir(parents=True, exist_ok=False)
            audio_path = voice_path / "prompt_audio.wav"

            try:
                # Save uploaded audio to a temporary file with the original
                # extension so pydub can detect the format.
                original_name = prompt_audio.filename or "upload.bin"
                suffix = Path(original_name).suffix or ".bin"
                tmp_path = voice_path / f"upload_{int(time.time())}{suffix}"
                with open(tmp_path, "wb") as f:
                    f.write(content)

                # Convert/resample to WAV at the model sample rate, mono
                audio = AudioSegment.from_file(tmp_path)
                audio = audio.set_frame_rate(settings.sample_rate).set_channels(1).set_sample_width(2)
                audio.export(str(audio_path), format="wav")

                os.remove(tmp_path)
            except Exception as e:
                shutil.rmtree(voice_path, ignore_errors=True)
                logger.error(f"Failed to process voice upload: {e}")
                raise HTTPException(status_code=400, detail=f"Could not process audio file: {e}")

            created_at = datetime.now(timezone.utc).isoformat()
            metadata = {
                "voice_id": safe_id,
                "name": name,
                "prompt_text": prompt_text,
                "created_at": created_at,
            }
            with open(_voice_meta_path(voice_path), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            voice = _load_voice_from_disk(voice_path)
            if voice:
                voices[safe_id] = voice

            return {"voice_id": safe_id, "name": name, "created_at": created_at}

    @app.get("/v1/voices")
    def list_voices(_=Depends(verify_auth())):
        return {
            "voices": [
                {
                    "voice_id": v.voice_id,
                    "name": v.name,
                    "created_at": v.created_at,
                }
                for v in voices.values()
            ]
        }

    @app.get("/v1/voices/{voice_id}")
    def get_voice(voice_id: str, _=Depends(verify_auth())):
        if voice_id not in voices:
            raise HTTPException(status_code=404, detail="Voice not found")
        v = voices[voice_id]
        return {
            "voice_id": v.voice_id,
            "name": v.name,
            "prompt_text": v.prompt_text,
            "created_at": v.created_at,
        }

    @app.delete("/v1/voices/{voice_id}")
    async def delete_voice(voice_id: str, _=Depends(verify_auth("admin"))):
        async with voice_lock:
            if voice_id not in voices:
                raise HTTPException(status_code=404, detail="Voice not found")
            voice_path = voices[voice_id].path
            del voices[voice_id]
            shutil.rmtree(voice_path, ignore_errors=True)
        return {"deleted": True, "voice_id": voice_id}

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Uvicorn target (`uvicorn api.server:app`). Settings are read from the
# GLM_TTS_* environment here, once, for the production process.
app = create_app(Settings())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=app.state.settings.port, log_level="info")
