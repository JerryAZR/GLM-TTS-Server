"""
FastAPI OpenAI-compatible TTS server for GLM-TTS.

Endpoints:
  - GET  /health
  - GET  /ready
  - GET  /v1/models
  - POST /v1/audio/speech
  - POST /v1/voices
  - GET  /v1/voices
  - GET  /v1/voices/{voice_id}
  - DELETE /v1/voices/{voice_id}
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field
from pydub import AudioSegment

from api.auth import load_authorized_keys, verify_auth, AUTH_KEYS_FILE
from glmtts_inference import generate_long, get_device_from_env, get_dtype_from_env, load_models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

MODEL_DIR = os.environ.get("GLM_TTS_MODEL_DIR", "")
if not MODEL_DIR:
    MODEL_DIR = "/workspace/ckpt" if Path("/workspace/ckpt").exists() else "ckpt"

VOICES_DIR = os.environ.get("GLM_TTS_VOICES_DIR", "")
if not VOICES_DIR:
    VOICES_DIR = "/workspace/voices" if Path("/workspace/voices").exists() else "voices"

DEVICE_STR = os.environ.get("GLM_TTS_DEVICE", "auto")
DTYPE_STR = os.environ.get("GLM_TTS_DTYPE", "float16")
PORT = int(os.environ.get("GLM_TTS_PORT", "8000"))
MOCK_INFERENCE = os.environ.get("GLM_TTS_MOCK_INFERENCE", "0") in ("1", "true", "True")

USE_PHONEME = os.environ.get("GLM_TTS_USE_PHONEME", "0") in ("1", "true", "True")
SAMPLE_RATE = int(os.environ.get("GLM_TTS_SAMPLE_RATE", "24000"))

# Image build stamp (CI passes --build-arg GIT_SHA=...; "unknown" otherwise).
GIT_SHA = os.environ.get("GLM_TTS_GIT_SHA", "unknown")
STARTED_AT = time.time()

# Lightweight request stats (in-memory; reset on restart). A "quick check"
# for the /status endpoint — anything heavier belongs in an SSH session.
STATS = {
    "speech_requests": 0,
    "failed_requests": 0,
    "audio_seconds_generated": 0.0,
    "last_generation_seconds": None,
}

# Max uploaded reference audio size (20 MB)
MAX_UPLOAD_BYTES = int(os.environ.get("GLM_TTS_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="GLM-TTS Server", version="0.1.0")

# ---------------------------------------------------------------------------
# Voice registry
# ---------------------------------------------------------------------------

class VoiceEntry(BaseModel):
    voice_id: str
    name: str
    prompt_text: str
    created_at: str
    path: Path


VOICE_REGISTRY: Dict[str, VoiceEntry] = {}
VOICE_LOCK = asyncio.Lock()


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


def scan_voices() -> None:
    """Scan the voices directory and populate the in-memory registry."""
    VOICE_REGISTRY.clear()
    voices_root = Path(VOICES_DIR)
    voices_root.mkdir(parents=True, exist_ok=True)
    for entry in voices_root.iterdir():
        if not entry.is_dir():
            continue
        voice = _load_voice_from_disk(entry)
        if voice:
            VOICE_REGISTRY[voice.voice_id] = voice
    logger.info(f"Loaded {len(VOICE_REGISTRY)} voice(s) from {voices_root}")


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    def __init__(self):
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

    def configure(self, device_str: str, dtype_str: str) -> None:
        self.device = get_device_from_env(device_str)
        # fp16 / bf16 are not supported for CPU inference
        if self.device.type == "cpu" and dtype_str.lower() in ("float16", "fp16", "half", "bfloat16", "bf16"):
            logger.warning(f"Device is CPU; forcing float32 instead of {dtype_str}")
            dtype_str = "float32"
        # Keep env in sync (after any override above) for code that reads it
        # via get_*_from_env(); get_dtype_from_env prefers the env value.
        os.environ["GLM_TTS_DEVICE"] = device_str
        os.environ["GLM_TTS_DTYPE"] = dtype_str
        self.dtype = get_dtype_from_env(dtype_str)
        logger.info(f"Inference engine configured: device={self.device}, dtype={self.dtype}")

    def load(self) -> None:
        if MOCK_INFERENCE:
            logger.info("Mock inference enabled; skipping model load.")
            self.ready = True
            return

        self.configure(DEVICE_STR, DTYPE_STR)
        logger.info(f"Loading models from {MODEL_DIR} ...")
        (
            self.frontend,
            self.text_frontend,
            self.speech_tokenizer,
            self.llm,
            self.flow,
        ) = load_models(
            use_phoneme=USE_PHONEME,
            sample_rate=SAMPLE_RATE,
            device=self.device,
            dtype=self.dtype,
            model_dir=MODEL_DIR,
        )
        self.ready = True
        logger.info("Models loaded successfully.")

    def _extract_prompt_features(self, voice: VoiceEntry):
        audio_path = voice.path / "prompt_audio.wav"
        prompt_text = self.text_frontend.text_normalize(voice.prompt_text)
        prompt_text_token = self.frontend._extract_text_token(prompt_text + " ")
        prompt_speech_token = self.frontend._extract_speech_token([str(audio_path)])
        speech_feat = self.frontend._extract_speech_feat(str(audio_path), sample_rate=SAMPLE_RATE)
        embedding = self.frontend._extract_spk_embedding(str(audio_path))

        # Keep audio features in fp32; only the LLM is cast to the configured dtype.
        speech_feat = speech_feat.to(device=self.device)
        embedding = embedding.to(device=self.device)

        cache_speech_token = [prompt_speech_token.squeeze().tolist()]
        flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(self.device)

        return prompt_text, prompt_text_token, cache_speech_token, flow_prompt_token, speech_feat, embedding

    def synthesize(self, text: str, voice_id: str) -> torch.Tensor:
        if MOCK_INFERENCE:
            # 1 second of silence-ish dummy sine wave at the target sample rate
            t = torch.linspace(0, 1, SAMPLE_RATE, dtype=torch.float32)
            wav = torch.sin(2 * 3.1415926 * 440 * t).unsqueeze(0)
            return wav

        if voice_id not in VOICE_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")

        voice = VOICE_REGISTRY[voice_id]
        audio_path = voice.path / "prompt_audio.wav"
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail=f"Voice audio for '{voice_id}' not found")

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
            text_info=[voice_id, text],
            cache=cache,
            embedding=embedding,
            device=self.device,
            flow_prompt_token=flow_prompt_token,
            speech_feat=speech_feat,
            use_phoneme=USE_PHONEME,
        )
        return tts_speech


ENGINE = InferenceEngine()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

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
# Startup
# ---------------------------------------------------------------------------

async def _load_engine() -> None:
    """Load models in a background thread so /health stays responsive."""
    try:
        await asyncio.to_thread(ENGINE.load)
    except Exception as exc:
        logger.exception("Inference engine failed to load")
        ENGINE.ready = False
        ENGINE.startup_error = str(exc)


@app.on_event("startup")
async def on_startup():
    load_authorized_keys(AUTH_KEYS_FILE)
    scan_voices()
    asyncio.create_task(_load_engine())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    payload = {"ready": ENGINE.ready, "mock": MOCK_INFERENCE}
    if not ENGINE.ready and ENGINE.startup_error:
        payload["error"] = ENGINE.startup_error
    return payload


@app.get("/version")
def version():
    """Public build stamp: which image revision is running."""
    return {"version": GIT_SHA, "mock": MOCK_INFERENCE}


@app.get("/status")
def server_status(_=Depends(verify_auth())):
    """Quick operational check: config, uptime, and request stats."""
    payload = {
        "version": GIT_SHA,
        "ready": ENGINE.ready,
        "mock": MOCK_INFERENCE,
        "device": str(ENGINE.device) if ENGINE.device is not None else DEVICE_STR,
        "dtype": str(ENGINE.dtype).replace("torch.", "") if ENGINE.dtype is not None else DTYPE_STR,
        "sample_rate": SAMPLE_RATE,
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "voices": len(VOICE_REGISTRY),
        "generating": ENGINE.lock.locked(),
        "stats": dict(STATS),
    }
    if not ENGINE.ready and ENGINE.startup_error:
        payload["startup_error"] = ENGINE.startup_error
    return payload


@app.get("/v1/models")
def list_models(_=Depends(verify_auth())):
    return {"data": [{"id": "glm-tts", "object": "model", "owned_by": "zai-org"}]}


class SpeechRequest(BaseModel):
    model: str = "glm-tts"
    input: str = Field(..., min_length=1, max_length=5000)
    voice: str
    response_format: str = "wav"  # wav or mp3
    speed: float = 1.0


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest, _=Depends(verify_auth(["speech:generate"]))):
    if not ENGINE.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not ready yet",
        )

    if req.response_format not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="response_format must be 'wav' or 'mp3'")

    if not req.input.strip():
        raise HTTPException(status_code=400, detail="input text cannot be empty")

    if req.voice not in VOICE_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Voice '{req.voice}' not found")

    STATS["speech_requests"] += 1
    async with ENGINE.lock:
        t0 = time.monotonic()
        try:
            wav = await asyncio.to_thread(ENGINE.synthesize, req.input, req.voice)
        except Exception:
            STATS["failed_requests"] += 1
            raise
        STATS["last_generation_seconds"] = round(time.monotonic() - t0, 2)
        STATS["audio_seconds_generated"] = round(
            STATS["audio_seconds_generated"] + wav.shape[-1] / SAMPLE_RATE, 2
        )
        wav_bytes = _tensor_to_wav_bytes(wav, SAMPLE_RATE)

    if req.response_format == "mp3":
        audio_bytes = _wav_to_mp3_bytes(wav_bytes)
        media_type = "audio/mpeg"
    else:
        audio_bytes = wav_bytes
        media_type = "audio/wav"

    return Response(content=audio_bytes, media_type=media_type)


@app.post("/v1/voices")
async def create_voice(
    name: str = Form(...),
    prompt_text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    voice_id: Optional[str] = Form(None),
    _=Depends(verify_auth(["voices:manage"])),
):
    if not voice_id:
        voice_id = f"voice_{uuid.uuid4().hex[:12]}"
    voice_id = voice_id.strip()
    if not voice_id:
        raise HTTPException(status_code=400, detail="voice_id cannot be empty")

    # Only allow simple IDs to avoid path traversal
    safe_id = "".join(c for c in voice_id if c.isalnum() or c in "-_")
    if not safe_id or safe_id != voice_id:
        raise HTTPException(status_code=400, detail="voice_id must be alphanumeric with '-' or '_'")

    content = await prompt_audio.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file exceeds {MAX_UPLOAD_BYTES} bytes ({len(content)} bytes uploaded)",
        )

    async with VOICE_LOCK:
        voice_path = Path(VOICES_DIR) / safe_id
        if voice_path.exists():
            raise HTTPException(status_code=409, detail=f"Voice '{safe_id}' already exists")

        voice_path.mkdir(parents=True, exist_ok=False)
        audio_path = voice_path / "prompt_audio.wav"

        try:
            # Save uploaded audio to a temporary file with the original extension so pydub can detect format
            original_name = prompt_audio.filename or "upload.bin"
            suffix = Path(original_name).suffix or ".bin"
            tmp_path = voice_path / f"upload_{int(time.time())}{suffix}"
            with open(tmp_path, "wb") as f:
                f.write(content)

            # Convert/resample to WAV at the model sample rate, mono
            audio = AudioSegment.from_file(tmp_path)
            audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
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
            VOICE_REGISTRY[safe_id] = voice

        return {"voice_id": safe_id, "name": name, "created_at": created_at}


@app.get("/v1/voices")
def list_voices(_=Depends(verify_auth(["voices:read"]))):
    return {
        "voices": [
            {
                "voice_id": v.voice_id,
                "name": v.name,
                "created_at": v.created_at,
            }
            for v in VOICE_REGISTRY.values()
        ]
    }


@app.get("/v1/voices/{voice_id}")
def get_voice(voice_id: str, _=Depends(verify_auth(["voices:read"]))):
    if voice_id not in VOICE_REGISTRY:
        raise HTTPException(status_code=404, detail="Voice not found")
    v = VOICE_REGISTRY[voice_id]
    return {
        "voice_id": v.voice_id,
        "name": v.name,
        "prompt_text": v.prompt_text,
        "created_at": v.created_at,
    }


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str, _=Depends(verify_auth(["voices:manage"]))):
    async with VOICE_LOCK:
        if voice_id not in VOICE_REGISTRY:
            raise HTTPException(status_code=404, detail="Voice not found")
        voice_path = VOICE_REGISTRY[voice_id].path
        del VOICE_REGISTRY[voice_id]
        shutil.rmtree(voice_path, ignore_errors=True)
    return {"deleted": True, "voice_id": voice_id}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
