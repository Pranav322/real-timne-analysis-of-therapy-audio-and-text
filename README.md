## Backend - Real-Time Therapy Session MVP

Lightweight FastAPI service that powers the multimodal analysis demo described in the paper. It streams audio/text turns from the Next.js frontend to external AI APIs (OpenAI Whisper + Hugging Face sentiment) and merges them with a simple prosody heuristic to expose therapist-friendly insights.

### 1. Prerequisites

- Python 3.13 (managed automatically via `uv`)
- `uv` CLI (already used to bootstrap this project)
- **Audio transcription options**
  - `OPENAI_API_KEY` – enables OpenAI Whisper API (`TRANSCRIPTION_PROVIDER=openai`, default).
  - `TRANSCRIPTION_PROVIDER=local` – enables the bundled offline Vosk pipeline. Requires `ffmpeg` installed (`sudo apt install ffmpeg`) and ~50 MB disk space for the `vosk-model-small-en-us-0.15` download (automatic on first run).
- **Sentiment**
  - `HF_API_KEY` – Hugging Face Inference token. Required to call the hosted sentiment model (`distilbert-base-uncased-finetuned-sst-2-english`). If you skip it, the service automatically falls back to the local VADER lexicon.
  - `GEMINI_API_KEY` – **recommended**. Enables Google Gemini (`gemini-2.0-flash` by default) for sentiment/emotion cues so you don't rely on OpenAI quota.
  - `USE_OPENAI_SENTIMENT=true` (optional) – reuse the same OpenAI API key for text sentiment.
  - `OPENAI_SENTIMENT_MODEL` / `GEMINI_SENTIMENT_MODEL` – override the chosen model ids.

### 2. Install dependencies

```bash
cd backend
uv sync
```

This creates/updates `.venv` and installs FastAPI, httpx, numpy, soundfile, etc.

### 3. Run locally

```bash
cd backend
uv run uvicorn main:app --reload --port 8000
```

Set environment variables in the same shell (or create a `.env` file):

```bash
export OPENAI_API_KEY=sk-...
export HF_API_KEY=hf_...
# optional overrides
export HF_SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
export OPENAI_WHISPER_MODEL=gpt-4o-mini-transcribe
```

### 4. Key endpoints

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET` | `/health` | Basic readiness probe |
| `POST` | `/sessions` | Creates a new analysis session and returns `session_id` |
| `GET` | `/sessions/{session_id}` | Full session summary (turns + timeline) |
| `POST` | `/sessions/{session_id}/messages` | Submit pre-transcribed text turns (patient/therapist) |
| `POST` | `/sessions/{session_id}/audio` | Upload/base64 live chunks for transcription + prosody analysis (OpenAI Whisper or offline Vosk) |
> Crisis safeguarding: regardless of sentiment provider, the backend scans for phrases like “I want to die” or “suicide” and auto-escalates the risk level to `critical`. This ensures realistic UI feedback even if an external sentiment API fails.

### 5. Request payloads

- **Text message**

```jsonc
{
  "speaker": "patient",
  "content": "I feel better than last week but still anxious before work."
}
```

- **Audio message**

```jsonc
{
  "speaker": "patient",
  "audio_base64": "<base64 encoded 16kHz WAV>",
  "mime_type": "audio/wav"
}
```

Responses include the transcript, sentiment scores, stress proxy, and fused insight (`engagement`, `stress`, `risk`, recommendations). The frontend can poll `GET /sessions/{id}` for historical turns to draw charts.

### 6. Architecture snapshot

- **External AI**: OpenAI Whisper for ASR, Hugging Face sentiment for emotion labels (falls back to an offline VADER lexicon if the API is unavailable). Audio can also remain entirely offline by switching to the Vosk-based transcriber.
- **Signal Processing**: `ProsodyAnalyzer` derives RMS + zero-crossing features to flag calm/elevated/high stress.
- **Insight Engine**: Rule-based fusion computing risk + engagement suggestions suitable for demos.
- **Storage**: In-memory `SessionStore` (no DB). Safe for college MVP; swap with Redis/Postgres later if persistence is needed.

### 7. Next steps

- Wire the Next.js frontend to these endpoints via REST/WebSocket wrappers.
- Add authentication/consent gates before enabling recording.
- Persist sessions to SQLite/Postgres if longer demos are required.

