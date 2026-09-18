import os
import json
import uuid
import base64
import asyncio
import subprocess
from urllib.parse import quote
from typing import Literal, Any

import websockets
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

PALABRA_API_KEY = os.environ["PALABRA_API_KEY"]
ADAPTER_API_KEY = os.environ["ADAPTER_API_KEY"]

PALABRA_WS_URL = os.getenv(
    "PALABRA_WS_URL",
    "wss://stream.palabra.ai/tts-api/v1/text-to-speech/stream"
)

# European Spanish. Change to "es-la" for Latin-American Spanish.
PALABRA_LANGUAGE = os.getenv("PALABRA_LANGUAGE", "es-eu")
PALABRA_MODEL = os.getenv("PALABRA_MODEL", "auto")

# Standard Palabra voice; no cloning needed.
PALABRA_VOICE = os.getenv("PALABRA_VOICE", "default_high")

# Palabra supports 8 kHz to 48 kHz.
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "24000"))

# Safety limit for one OpenMAIC request.
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "20000"))


app = FastAPI(
    title="OpenMAIC -> Palabra.ai TTS Adapter",
    version="1.0.0",
)


# -------------------------------------------------------------------
# OpenAI-compatible request shape
# -------------------------------------------------------------------

class SpeechRequest(BaseModel):
    model: str = "palabra-tts"
    input: str
    voice: Any = "default"
    response_format: Literal[
        "mp3", "wav", "pcm", "flac", "aac", "opus"
    ] = "mp3"
    speed: float = 1.0


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def check_auth(authorization: str | None) -> None:
    expected = f"Bearer {ADAPTER_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def split_text(text: str, max_chars: int = 1000) -> list[str]:
    """
    Palabra accepts up to 1024 characters per text message.
    We use 1000 to leave some margin and try to split at punctuation.
    """
    text = " ".join(text.strip().split())
    if not text:
        return []

    chunks: list[str] = []

    while len(text) > max_chars:
        cut = max(
            text.rfind(". ", 0, max_chars),
            text.rfind("? ", 0, max_chars),
            text.rfind("! ", 0, max_chars),
            text.rfind("; ", 0, max_chars),
        )

        if cut < max_chars // 3:
            cut = text.rfind(" ", 0, max_chars)

        if cut <= 0:
            cut = max_chars
        else:
            # Keep the punctuation mark when present.
            cut += 1

        part = text[:cut].strip()
        if part:
            chunks.append(part)

        text = text[cut:].strip()

    if text:
        chunks.append(text)

    return chunks


def pcm_to_output(pcm: bytes, output_format: str) -> tuple[bytes, str]:
    """
    Convert raw 16-bit little-endian PCM from Palabra to the format
    requested by OpenMAIC/OpenAI-compatible clients.
    """
    if output_format == "pcm":
        return pcm, "application/octet-stream"

    configs = {
        "mp3": (["-c:a", "libmp3lame", "-f", "mp3"], "audio/mpeg"),
        "wav": (["-c:a", "pcm_s16le", "-f", "wav"], "audio/wav"),
        "flac": (["-c:a", "flac", "-f", "flac"], "audio/flac"),
        "aac": (["-c:a", "aac", "-f", "adts"], "audio/aac"),
        "opus": (["-c:a", "libopus", "-f", "ogg"], "audio/ogg"),
    }

    ffmpeg_args, media_type = configs[output_format]

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-i", "pipe:0",
        *ffmpeg_args,
        "pipe:1",
    ]

    result = subprocess.run(
        command,
        input=pcm,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode("utf-8", errors="ignore")
        )

    return result.stdout, media_type


async def synthesize_chunk(ws, text: str, speed: float) -> bytes:
    generation_id = uuid.uuid4().hex
    speed = max(0.1, min(float(speed), 2.0))

    await ws.send(json.dumps({
        "type": "text",
        "text": text,
        "generation_id": generation_id,
        "is_eos": True,
        "voice_options": {
            "speed": speed
        }
    }))

    audio = bytearray()

    while True:
        raw = await ws.recv()
        message = json.loads(raw)

        if message.get("message_type") == "error":
            data = message.get("data") or {}
            raise RuntimeError(
                f"Palabra error {data.get('code', '')}: "
                f"{data.get('desc', 'Unknown error')}"
            )

        if message.get("message_type") != "audio_chunk":
            continue

        data = message.get("data") or {}

        if data.get("generation_id") != generation_id:
            continue

        encoded = data.get("audio")
        if encoded:
            audio.extend(base64.b64decode(encoded))

        if data.get("last_chunk"):
            break

    return bytes(audio)


async def palabra_tts(text: str, speed: float) -> bytes:
    """
    Palabra dedicated Realtime TTS:
      connect -> init -> text -> audio_chunk
    We request PCM so multiple chunks can be safely concatenated before
    converting the complete result to MP3/WAV/etc.
    """
    url = f"{PALABRA_WS_URL}?token={quote(PALABRA_API_KEY, safe='')}"
    pcm = bytearray()

    async with websockets.connect(
        url,
        ping_interval=20,
        ping_timeout=30,
        max_size=None,
    ) as ws:
        await ws.send(json.dumps({
            "type": "init",
            "language": PALABRA_LANGUAGE,
            "model": PALABRA_MODEL,
            "voice_options": {
                "voice_id": PALABRA_VOICE,
                "speed": max(0.1, min(float(speed), 2.0)),
                "deaccent_strength": 1.0,
            },
            "output": {
                "format": "pcm",
                "sample_rate": SAMPLE_RATE,
            }
        }))

        for chunk in split_text(text):
            pcm.extend(await synthesize_chunk(ws, chunk, speed))

    return bytes(pcm)


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "OpenMAIC Palabra TTS Adapter",
        "status": "ok",
        "health": "/health",
        "speech": "/v1/audio/speech",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "provider": "palabra.ai",
        "language": PALABRA_LANGUAGE,
        "voice": PALABRA_VOICE,
    }


@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)):
    check_auth(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": "palabra-tts",
                "object": "model",
                "owned_by": "palabra.ai"
            }
        ]
    }


@app.post("/v1/audio/speech")
async def create_speech(
    request: SpeechRequest,
    authorization: str | None = Header(default=None),
):
    check_auth(authorization)

    if not request.input.strip():
        raise HTTPException(status_code=400, detail="input cannot be empty")

    if len(request.input) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"input exceeds {MAX_INPUT_CHARS} characters"
        )

    try:
        pcm = await asyncio.wait_for(
            palabra_tts(request.input, request.speed),
            timeout=180,
        )

        audio, media_type = await asyncio.to_thread(
            pcm_to_output,
            pcm,
            request.response_format,
        )

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Palabra TTS timeout")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return Response(
        content=audio,
        media_type=media_type,
        headers={"Cache-Control": "no-store"},
    )
