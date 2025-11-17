from __future__ import annotations

import base64

from fastapi import FastAPI, HTTPException
import logging
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import (
    AudioMessageRequest,
    SentimentResult,
    SessionCreateResponse,
    SessionSummary,
    StressResult,
    TextMessageRequest,
    TurnResponse,
)
from app.services import (
    CrisisDetector,
    GeminiService,
    HuggingFaceService,
    InsightEngine,
    LexiconSentiment,
    LocalVoskTranscriber,
    ProsodyAnalyzer,
)
from app.store import SessionNotFoundError, SessionStore
from fastapi import Response
import tempfile
from pathlib import Path

settings = get_settings()

app = FastAPI(
    title="Therapy Session Analyzer",
    version="0.1.0",
    description="College-level MVP for multimodal therapy session analysis.",
)

logger = logging.getLogger("uvicorn.error")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_store = SessionStore()
hf_service = HuggingFaceService(
    api_key=settings.hf_api_key,
    model=settings.sentiment_model,
)
insight_engine = InsightEngine()
prosody_analyzer = ProsodyAnalyzer()
crisis_detector = CrisisDetector()
lexicon_service = LexiconSentiment()
gemini_service = GeminiService(
    api_key=settings.gemini_api_key,
    model=settings.gemini_sentiment_model,
)
transcription_provider = settings.transcription_provider.lower()
local_transcriber: LocalVoskTranscriber | None = None
# Default to local transcriber unless explicitly configured otherwise.
if transcription_provider == "local":
    local_transcriber = LocalVoskTranscriber(
        model_name=settings.local_asr_model_name,
        model_url=settings.local_asr_model_url,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sessions", response_model=SessionCreateResponse)
async def create_session() -> SessionCreateResponse:
    session_id = await session_store.create_session()
    return SessionCreateResponse(session_id=session_id)


@app.get("/sessions/{session_id}", response_model=SessionSummary)
async def get_session(session_id: str) -> SessionSummary:
    try:
        return await session_store.get_summary(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post(
    "/sessions/{session_id}/messages",
    response_model=TurnResponse,
    summary="Submit a text message (already transcribed).",
)
async def add_text_message(
    session_id: str, payload: TextMessageRequest
) -> TurnResponse:
    try:
        await session_store.ensure_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    sentiment = await compute_sentiment(payload.content)
    crisis_signal = crisis_detector.detect(payload.content)
    if crisis_signal.triggered:
        sentiment.label = "negative"
        sentiment.score = min(sentiment.score, 0.25)
    insight = insight_engine.build_insight(
        sentiment=sentiment,
        stress=None,
        crisis_triggered=crisis_signal.triggered,
    )

    turn = await session_store.add_turn(
        session_id=session_id,
        speaker=payload.speaker,
        transcript=payload.content,
        sentiment=sentiment.model_dump(),
        stress=None,
        insight=insight,
    )
    return turn


@app.post(
    "/sessions/{session_id}/audio",
    response_model=TurnResponse,
    summary="Submit an audio snippet for ASR + emotion processing.",
)
async def add_audio_message(
    session_id: str, payload: AudioMessageRequest
) -> TurnResponse:
    try:
        await session_store.ensure_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    # Helpful debug logging to diagnose invalid audio / mime_type issues.
    logger.info(
        "Incoming audio request: session=%s mime_type=%s b64_len=%d",
        session_id,
        payload.mime_type,
        len(payload.audio_base64 or ""),
    )
    try:
        audio_bytes = base64.b64decode(payload.audio_base64)
    except Exception as exc:  # pragma: no cover - runtime validation
        logger.exception("Base64 decode failed for session=%s", session_id)
        raise HTTPException(status_code=400, detail="Invalid base64 audio payload")

    # Log first bytes to help identify format (will appear in server logs).
    try:
        logger.info(
            "Audio sample (first 32 bytes hex): %s",
            audio_bytes[:32].hex(),
        )
    except Exception:
        logger.debug("Unable to log audio sample bytes")

    # If the client is sending chunked WebM/Opus blobs (MediaRecorder with
    # timeslice), we need to assemble them server-side. If chunk_index is
    # present and is_last is False, append the bytes to a temp file and return
    # 202 Accepted so the client can continue sending chunks. When is_last is
    # True, append the final chunk and proceed with transcription on the
    # assembled file.
    chunk_dir = Path(tempfile.gettempdir()) / "therapy_audio_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_file = chunk_dir / f"{session_id}.upload"

    if payload.chunk_index is not None:
        # Append chunk to the session file.
        try:
            with chunk_file.open("ab") as fh:
                fh.write(audio_bytes)
        except OSError as exc:
            logger.exception("Failed to append audio chunk for %s", session_id)
            raise HTTPException(status_code=500, detail="Unable to store audio chunk")

        if not payload.is_last:
            # Intermediate chunk — acknowledge and return early.
            return Response(status_code=202)

        # Final chunk: read the assembled file as the full audio payload.
        try:
            with chunk_file.open("rb") as fh:
                audio_bytes = fh.read()
        finally:
            try:
                chunk_file.unlink(missing_ok=True)
            except Exception:
                logger.debug("Could not delete chunk file %s", chunk_file)

    try:
        transcript = await transcribe_audio(
            audio_base64=payload.audio_base64,
            audio_bytes=audio_bytes,
            mime_type=payload.mime_type,
        )
    except RuntimeError as err:
        raise HTTPException(status_code=400, detail=str(err))

    sentiment = await compute_sentiment(transcript)
    crisis_signal = crisis_detector.detect(transcript)
    if crisis_signal.triggered:
        sentiment.label = "negative"
        sentiment.score = min(sentiment.score, 0.2)
    # Prosody analyzer expects readable PCM/WAV data (libsndfile). Convert
    # compressed inputs (webm/opus, mp3, etc.) to a mono 16k WAV first so
    # soundfile can parse them reliably.
    try:
        wav_bytes = audio_bytes
        if local_transcriber is not None:
            # run conversion in threadpool to avoid blocking the event loop
            wav_bytes = await run_in_threadpool(
                local_transcriber.convert_to_wav_bytes, audio_bytes, payload.mime_type
            )
        prosody = prosody_analyzer.analyze(wav_bytes)
    except RuntimeError as err:
        # surface a clear 400-level error for conversion/read failures
        raise HTTPException(status_code=400, detail=f"Audio processing error: {err}")
    stress = StressResult(
        label=prosody.label,
        score=prosody.score,
        duration_seconds=prosody.duration_seconds,
    )

    insight = insight_engine.build_insight(
        sentiment=sentiment,
        stress=stress,
        crisis_triggered=crisis_signal.triggered,
    )

    turn = await session_store.add_turn(
        session_id=session_id,
        speaker=payload.speaker,
        transcript=transcript,
        sentiment=sentiment.model_dump(),
        stress=stress.model_dump(),
        insight=insight,
    )
    return turn


async def compute_sentiment(text: str) -> SentimentResult:
    cleaned = text.strip()
    if not cleaned:
        return SentimentResult(label="neutral", score=0.0)

    # Prefer Hugging Face transformer when token is available.
    if settings.hf_api_key:
        try:
            sentiment_data = await hf_service.sentiment(cleaned)
            return _normalize_sentiment_payload(sentiment_data)
        except RuntimeError:
            pass

    # Optional Gemini classifier.
    if gemini_service.enabled:
        try:
            sentiment_data = await gemini_service.sentiment(cleaned)
            return _normalize_sentiment_payload(sentiment_data)
        except RuntimeError:
            pass

    # Optional: OpenAI sentiment has been removed from the default flow. If you
    # absolutely need OpenAI-based sentiment, re-enable it manually. For now,
    # we prefer Hugging Face -> Gemini -> lexical fallback.

    # Offline fallback (VADER-based).
    lexical = lexicon_service.score(cleaned)
    return _normalize_sentiment_payload(lexical)


def _normalize_sentiment_payload(payload: dict[str, float | str]) -> SentimentResult:
    score = float(payload.get("score", 0.0))
    label = str(payload.get("label", "neutral")).lower()
    if label not in {"positive", "neutral", "negative"}:
        if score > 0.6:
            label = "positive"
        elif score < 0.4:
            label = "negative"
        else:
            label = "neutral"
    return SentimentResult(label=label, score=round(score, 3))


async def transcribe_audio(
    audio_base64: str,
    audio_bytes: bytes,
    mime_type: str,
) -> str:
    # Use the local Vosk transcriber by default. OpenAI-based transcription has
    # been removed from the default flow to avoid unauthorized API calls.
    if local_transcriber:
        return await run_in_threadpool(
            local_transcriber.transcribe, audio_bytes, mime_type
        )
    raise RuntimeError(
        "Transcription requires TRANSCRIPTION_PROVIDER=local with ffmpeg installed."
    )
