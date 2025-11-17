from __future__ import annotations

import base64

from fastapi import FastAPI, HTTPException
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
    OpenAIService,
    ProsodyAnalyzer,
)
from app.store import SessionNotFoundError, SessionStore

settings = get_settings()

app = FastAPI(
    title="Therapy Session Analyzer",
    version="0.1.0",
    description="College-level MVP for multimodal therapy session analysis.",
)

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
openai_service = OpenAIService(
    api_key=settings.openai_api_key,
    model=settings.openai_whisper_model,
    text_model=settings.openai_sentiment_model,
)
crisis_detector = CrisisDetector()
lexicon_service = LexiconSentiment()
gemini_service = GeminiService(
    api_key=settings.gemini_api_key,
    model=settings.gemini_sentiment_model,
)
transcription_provider = settings.transcription_provider.lower()
use_openai_transcriber = bool(
    settings.openai_api_key and transcription_provider == "openai"
)
local_transcriber: LocalVoskTranscriber | None = None
if transcription_provider == "local" or not use_openai_transcriber:
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

    audio_bytes = base64.b64decode(payload.audio_base64)
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
    prosody = prosody_analyzer.analyze(audio_bytes)
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

    # Optional OpenAI classifier.
    if settings.use_openai_sentiment and settings.openai_api_key:
        try:
            sentiment_data = await openai_service.classify_sentiment(cleaned)
            return _normalize_sentiment_payload(sentiment_data)
        except RuntimeError:
            pass

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
    if use_openai_transcriber:
        try:
            return await openai_service.transcribe(
                audio_base64=audio_base64,
                mime_type=mime_type,
            )
        except RuntimeError:
            if not local_transcriber:
                raise
    if local_transcriber:
        return await run_in_threadpool(
            local_transcriber.transcribe, audio_bytes, mime_type
        )
    raise RuntimeError(
        "Transcription requires either OPENAI_API_KEY or TRANSCRIPTION_PROVIDER=local with ffmpeg installed."
    )
