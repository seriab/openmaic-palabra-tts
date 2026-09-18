# OpenMAIC <-> Palabra.ai Audio Adapter

OpenAI-compatible adapter for Palabra.ai:

- TTS: `POST /v1/audio/speech`
- ASR/STT: `POST /v1/audio/transcriptions`
- Models: `GET /v1/models`
- Health: `GET /health`

## Dokploy

Internal port: `8000`.

Required environment variables:

```env
PALABRA_API_KEY=YOUR_REAL_PALABRA_KEY
ADAPTER_API_KEY=YOUR_OWN_LONG_RANDOM_SECRET

# TTS
PALABRA_LANGUAGE=es-la
PALABRA_MODEL=auto
PALABRA_VOICE=default_high

# ASR
PALABRA_ASR_LANGUAGE=es
```

After updating from the older TTS-only version, use **Rebuild** in Dokploy because `requirements.txt` now adds `python-multipart`.

## OpenMAIC TTS

- Base URL: `https://YOUR-DOMAIN/v1`
- Model: `palabra-tts`
- API key: `ADAPTER_API_KEY`

## OpenMAIC ASR

- Base URL: `https://YOUR-DOMAIN/v1`
- Model: `palabra-asr`
- API key: `ADAPTER_API_KEY`
- Language: `es`

## ASR test

```bash
curl -X POST "https://YOUR-DOMAIN/v1/audio/transcriptions" \
  -H "Authorization: Bearer YOUR_ADAPTER_API_KEY" \
  -F "file=@prueba.wav" \
  -F "model=palabra-asr" \
  -F "language=es"
```

Expected:

```json
{"text":"Texto reconocido..."}
```
