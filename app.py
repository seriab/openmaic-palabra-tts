import os, json, uuid, base64, asyncio, subprocess
from urllib.parse import quote, urlencode
from typing import Literal, Any

import websockets
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse, PlainTextResponse
from pydantic import BaseModel

PALABRA_API_KEY = os.environ["PALABRA_API_KEY"]
ADAPTER_API_KEY = os.environ["ADAPTER_API_KEY"]

# TTS
PALABRA_TTS_WS_URL = os.getenv(
    "PALABRA_TTS_WS_URL",
    "wss://stream.palabra.ai/tts-api/v1/text-to-speech/stream",
)
PALABRA_LANGUAGE = os.getenv("PALABRA_LANGUAGE", "es-la")
PALABRA_MODEL = os.getenv("PALABRA_MODEL", "auto")
PALABRA_VOICE = os.getenv("PALABRA_VOICE", "default_high")
TTS_SAMPLE_RATE = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "20000"))

# ASR / STT
PALABRA_ASR_WS_URL = os.getenv(
    "PALABRA_ASR_WS_URL",
    "wss://stream.palabra.ai/asr/v1/speech-to-text/stream",
)
PALABRA_ASR_LANGUAGE = os.getenv("PALABRA_ASR_LANGUAGE", "es")
ASR_SAMPLE_RATE = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
ASR_CHUNK_MS = int(os.getenv("ASR_CHUNK_MS", "320"))
MAX_AUDIO_MB = int(os.getenv("MAX_AUDIO_MB", "25"))
SUPPORTED_ASR_LANGUAGES = {
    "ar", "de", "en", "es", "fr", "hi", "it", "ja", "ko", "nl", "pt", "ru", "zh", "auto"
}

app = FastAPI(title="OpenMAIC <-> Palabra.ai Audio Adapter", version="2.0.0")


def check_auth(authorization: str | None) -> None:
    if authorization != f"Bearer {ADAPTER_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


class SpeechRequest(BaseModel):
    model: str = "palabra-tts"
    input: str
    voice: Any = "default"
    response_format: Literal["mp3", "wav", "pcm", "flac", "aac", "opus"] = "mp3"
    speed: float = 1.0


def split_text(text: str, max_chars: int = 1000) -> list[str]:
    text = " ".join(text.strip().split())
    if not text:
        return []
    chunks = []
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
            cut += 1
        part = text[:cut].strip()
        if part:
            chunks.append(part)
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


def pcm_to_output(pcm: bytes, output_format: str) -> tuple[bytes, str]:
    if output_format == "pcm":
        return pcm, "application/octet-stream"
    configs = {
        "mp3": (["-c:a", "libmp3lame", "-f", "mp3"], "audio/mpeg"),
        "wav": (["-c:a", "pcm_s16le", "-f", "wav"], "audio/wav"),
        "flac": (["-c:a", "flac", "-f", "flac"], "audio/flac"),
        "aac": (["-c:a", "aac", "-f", "adts"], "audio/aac"),
        "opus": (["-c:a", "libopus", "-f", "ogg"], "audio/ogg"),
    }
    args, media_type = configs[output_format]
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(TTS_SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
        *args, "pipe:1",
    ]
    result = subprocess.run(cmd, input=pcm, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))
    return result.stdout, media_type


async def synthesize_tts_chunk(ws, text: str, speed: float) -> bytes:
    generation_id = uuid.uuid4().hex
    speed = max(0.1, min(float(speed), 2.0))
    await ws.send(json.dumps({
        "type": "text",
        "text": text,
        "generation_id": generation_id,
        "is_eos": True,
        "voice_options": {"speed": speed},
    }))
    audio = bytearray()
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("message_type") == "error":
            data = msg.get("data") or {}
            raise RuntimeError(f"Palabra TTS error {data.get('code', '')}: {data.get('desc', 'Unknown error')}")
        if msg.get("message_type") != "audio_chunk":
            continue
        data = msg.get("data") or {}
        if data.get("generation_id") != generation_id:
            continue
        if data.get("audio"):
            audio.extend(base64.b64decode(data["audio"]))
        if data.get("last_chunk"):
            break
    return bytes(audio)


async def palabra_tts(text: str, speed: float) -> bytes:
    url = f"{PALABRA_TTS_WS_URL}?token={quote(PALABRA_API_KEY, safe='')}"
    pcm = bytearray()
    async with websockets.connect(url, ping_interval=20, ping_timeout=30, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "init",
            "language": PALABRA_LANGUAGE,
            "model": PALABRA_MODEL,
            "voice_options": {
                "voice_id": PALABRA_VOICE,
                "speed": max(0.1, min(float(speed), 2.0)),
                "deaccent_strength": 1.0,
            },
            "output": {"format": "pcm", "sample_rate": TTS_SAMPLE_RATE},
        }))
        for chunk in split_text(text):
            pcm.extend(await synthesize_tts_chunk(ws, chunk, speed))
    return bytes(pcm)


def normalize_asr_language(language: str | None) -> str:
    language = (language or PALABRA_ASR_LANGUAGE).strip().lower().replace("_", "-")
    if "-" in language:
        language = language.split("-", 1)[0]
    return language if language in SUPPORTED_ASR_LANGUAGES else PALABRA_ASR_LANGUAGE


def audio_to_pcm16_mono_16k(audio: bytes) -> bytes:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
        "-vn", "-ac", "1", "-ar", str(ASR_SAMPLE_RATE),
        "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
    ]
    result = subprocess.run(cmd, input=audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError("Could not decode audio: " + result.stderr.decode("utf-8", errors="ignore"))
    return result.stdout


async def palabra_asr(pcm: bytes, language: str) -> str:
    params = {
        "token": PALABRA_API_KEY,
        "language": language,
        "format": "pcm_s16le",
        "sample_rate": str(ASR_SAMPLE_RATE),
        "finalization_mode": "manual",
    }
    url = f"{PALABRA_ASR_WS_URL}?{urlencode(params)}"
    chunk_bytes = int(ASR_SAMPLE_RATE * (ASR_CHUNK_MS / 1000.0) * 2)
    segments: dict[str, tuple[float, str]] = {}

    async with websockets.connect(url, ping_interval=20, ping_timeout=30, max_size=None) as ws:
        async def sender():
            for pos in range(0, len(pcm), chunk_bytes):
                chunk = pcm[pos:pos + chunk_bytes]
                if not chunk:
                    break
                await ws.send(chunk)
                await asyncio.sleep(ASR_CHUNK_MS / 1000.0)
            await ws.send(json.dumps({"message_type": "finalize"}))

        async def receiver() -> str:
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("message_type") != "transcription":
                    continue
                segment = msg.get("segment") or {}
                text = (segment.get("text") or "").strip()
                tid = msg.get("transcription_id") or uuid.uuid4().hex
                start_time = float(segment.get("start_time") or 0.0)
                got_fin = "<fin>" in text
                clean = text.replace("<fin>", "").strip()
                if clean:
                    segments[tid] = (start_time, clean)
                if got_fin:
                    return " ".join(
                        part for _, part in sorted(segments.values(), key=lambda x: x[0]) if part
                    ).strip()

        _, transcript = await asyncio.gather(sender(), receiver())
        return transcript


@app.get("/")
async def root():
    return {
        "service": "OpenMAIC Palabra Audio Adapter",
        "version": "2.0.0",
        "status": "ok",
        "tts": "/v1/audio/speech",
        "asr": "/v1/audio/transcriptions",
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "provider": "palabra.ai",
        "tts": {"language": PALABRA_LANGUAGE, "voice": PALABRA_VOICE},
        "asr": {"language": PALABRA_ASR_LANGUAGE, "sample_rate": ASR_SAMPLE_RATE},
    }


@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)):
    check_auth(authorization)
    return {
        "object": "list",
        "data": [
            {"id": "palabra-tts", "object": "model", "owned_by": "palabra.ai"},
            {"id": "palabra-asr", "object": "model", "owned_by": "palabra.ai"},
        ],
    }


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest, authorization: str | None = Header(default=None)):
    check_auth(authorization)
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="input cannot be empty")
    if len(request.input) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=413, detail=f"input exceeds {MAX_INPUT_CHARS} characters")
    try:
        pcm = await asyncio.wait_for(palabra_tts(request.input, request.speed), timeout=180)
        audio, media_type = await asyncio.to_thread(pcm_to_output, pcm, request.response_format)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Palabra TTS timeout")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return Response(content=audio, media_type=media_type, headers={"Cache-Control": "no-store"})


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("palabra-asr"),
    language: str | None = Form(None),
    response_format: str = Form("json"),
    prompt: str | None = Form(None),
    temperature: float | None = Form(None),
    authorization: str | None = Header(default=None),
):
    check_auth(authorization)
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Audio file is empty")
    if len(audio) > MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Audio exceeds {MAX_AUDIO_MB} MB")

    asr_language = normalize_asr_language(language)
    try:
        pcm = await asyncio.to_thread(audio_to_pcm16_mono_16k, audio)
        transcript = await asyncio.wait_for(palabra_asr(pcm, asr_language), timeout=600)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Palabra ASR timeout")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    if response_format == "text":
        return PlainTextResponse(transcript)
    if response_format == "verbose_json":
        return JSONResponse({"text": transcript, "language": asr_language, "segments": []})
    return JSONResponse({"text": transcript})
