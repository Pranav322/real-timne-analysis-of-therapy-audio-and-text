from __future__ import annotations

import base64
import json
from typing import Optional

import httpx


class OpenAIService:
    """Thin wrapper around OpenAI Whisper endpoints."""

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "gpt-4o-mini-transcribe",
        text_model: str = "gpt-4o-mini",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.text_model = text_model
        self.timeout = timeout
        self._base_url = "https://api.openai.com/v1"

    async def transcribe(
        self, audio_base64: str, mime_type: str = "audio/wav"
    ) -> str:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Provide an API key to enable transcription."
            )
        audio_bytes = base64.b64decode(audio_base64)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        files = {
            "file": ("audio", audio_bytes, mime_type),
        }
        data = {"model": self.model}
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=self.timeout
        ) as client:
            response = await client.post("/audio/transcriptions", headers=headers, data=data, files=files)
            response.raise_for_status()
            payload = response.json()
            return payload.get("text", "").strip()

    async def classify_sentiment(self, text: str) -> dict[str, float | str]:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for sentiment when USE_OPENAI_SENTIMENT=true."
            )
        prompt = (
            "Return a JSON object with keys label (positive, neutral, negative) and "
            "score (0-1 confidence). Text:\n"
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.text_model,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "You detect sentiment in therapy transcripts. Only respond with JSON.",
                },
                {"role": "user", "content": f"{prompt}{text.strip()}"},
            ],
            "temperature": 0,
        }
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=self.timeout
        ) as client:
            response = await client.post("/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        return json.loads(content)



