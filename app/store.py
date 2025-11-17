from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .schemas import InsightPayload, SessionSummary, SessionTimelinePoint, SpeakerRole, TurnResponse


class SessionNotFoundError(KeyError):
    """Raised when the requested session id is unknown."""


class SessionStore:
    """In-memory session storage suitable for demos."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        session_id = uuid4().hex
        payload = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc),
            "metadata": metadata or {},
            "turns": [],
            "timeline": [],
        }
        async with self._lock:
            self._sessions[session_id] = payload
        return session_id

    async def _get_session(self, session_id: str) -> Dict[str, Any]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise SessionNotFoundError(session_id)
            return session

    async def add_turn(
        self,
        session_id: str,
        speaker: SpeakerRole,
        transcript: str,
        sentiment: Optional[Dict[str, Any]] = None,
        stress: Optional[Dict[str, Any]] = None,
        insight: Optional[InsightPayload] = None,
    ) -> TurnResponse:
        session = await self._get_session(session_id)
        turn_id = uuid4().hex
        now = datetime.now(timezone.utc)
        turn_payload = {
            "turn_id": turn_id,
            "session_id": session_id,
            "speaker": speaker,
            "transcript": transcript,
            "created_at": now,
            "sentiment": sentiment,
            "stress": stress,
            "insight": insight.model_dump() if insight else None,
        }
        timeline_point = {
            "turn_id": turn_id,
            "timestamp": now,
            "sentiment_score": sentiment.get("score") if sentiment else None,
            "stress_score": stress.get("score") if stress else None,
            "risk": insight.risk if insight else None,
        }
        async with self._lock:
            session["turns"].append(turn_payload)
            session["timeline"].append(timeline_point)
        return TurnResponse(**turn_payload)

    async def get_summary(self, session_id: str) -> SessionSummary:
        session = await self._get_session(session_id)
        return SessionSummary(
            session_id=session_id,
            created_at=session["created_at"],
            turns=[TurnResponse(**turn) for turn in session["turns"]],
            timeline=[SessionTimelinePoint(**point) for point in session["timeline"]],
        )

    async def ensure_session(self, session_id: str) -> None:
        await self._get_session(session_id)


