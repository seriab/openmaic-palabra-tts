# OpenMAIC -> Palabra.ai TTS Adapter

Small FastAPI service that exposes an OpenAI-compatible TTS endpoint:

`POST /v1/audio/speech`

and forwards the text to Palabra.ai Realtime TTS.

## Dokploy (Drop) deployment

1. Create a new **Application** in Dokploy.
2. In **Provider**, choose **Drop** and upload the ZIP.
3. In **Build Type**, choose **Dockerfile**.
4. Dockerfile path: `Dockerfile`
5. Docker context: `.`
6. In **Environment**, add:

```env
PALABRA_API_KEY=YOUR_REAL_PALABRA_KEY
ADAPTER_API_KEY=YOUR_OWN_LONG_RANDOM_SECRET
PALABRA_LANGUAGE=es-eu
PALABRA_MODEL=auto
PALABRA_VOICE=default_high
```

7. Deploy.
8. Add a domain in Dokploy pointing to internal port `8000`, for example:
   `tts.example.com`
9. Test:
   `https://tts.example.com/health`

Expected JSON:

```json
{
  "status": "ok",
  "provider": "palabra.ai",
  "language": "es-eu",
  "voice": "default_high"
}
```

## Test TTS

```bash
curl -X POST "https://tts.example.com/v1/audio/speech" \
  -H "Authorization: Bearer YOUR_ADAPTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "palabra-tts",
    "input": "Hola. Bienvenidos a la clase de análisis de datos.",
    "voice": "default",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output prueba.mp3
```

## OpenMAIC custom TTS

Configure:

- Name: `Palabra AI`
- Base URL: `https://tts.example.com/v1`
- Default model: `palabra-tts`
- Requires API key: enabled
- API key: use `ADAPTER_API_KEY`, NOT your real Palabra key.

The real `PALABRA_API_KEY` stays only in Dokploy.
