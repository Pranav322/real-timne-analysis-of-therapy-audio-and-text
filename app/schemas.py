from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class SpeakerRole(str, Enum):
    patient = "patient"
    therapist = "therapist"
    system = "system"


class SessionCreateResponse(BaseModel):
    session_id: str


class TextMessageRequest(BaseModel):
    speaker: SpeakerRole = Field(..., description="Who produced the utterance.")
    content: str = Field(..., min_length=1, description="Transcribed textual content.")


class AudioMessageRequest(BaseModel):
    speaker: SpeakerRole
    audio_base64: str = Field(
        ..., description="Base64 encoded audio chunk (16 kHz wav/ogg)."
    )
    mime_type: str = Field(
        default="audio/wav", description="Mime type hint for the uploaded audio."
    )


class SentimentResult(BaseModel):
    label: str
    score: float


class StressResult(BaseModel):
    label: str
    score: float
    duration_seconds: float


class InsightPayload(BaseModel):
    engagement: str
    stress: str
    risk: str
    recommendations: List[str]


class TurnResponse(BaseModel):
    turn_id: str
    session_id: str
    speaker: SpeakerRole
    transcript: str
    created_at: datetime
    sentiment: Optional[SentimentResult]
    stress: Optional[StressResult]
    insight: Optional[InsightPayload]


class SessionTimelinePoint(BaseModel):
    turn_id: str
    timestamp: datetime
    sentiment_score: Optional[float]
    stress_score: Optional[float]
    risk: Optional[str]


class SessionSummary(BaseModel):
    session_id: str
    created_at: datetime
    turns: List[TurnResponse]
    timeline: List[SessionTimelinePoint]



