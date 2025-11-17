from __future__ import annotations

import json
from typing import Optional

import google.generativeai as genai
from fastapi.concurrency import run_in_threadpool


class GeminiService:
    """Wraps Google Gemini API for sentiment classification."""

    def __init__(self, api_key: Optional[str], model: str) -> None:
        self.api_key = api_key
        self.model_id = model
        self._model: Optional[genai.GenerativeModel] = None

        if api_key:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(model)

    @property
    def enabled(self) -> bool:
        return self._model is not None

    async def sentiment(self, text: str) -> dict[str, float | str]:
        if not self.enabled:
            raise RuntimeError("Gemini client is not configured (missing GEMINI_API_KEY).")

        prompt = (
            "You analyze therapy transcripts. Return JSON with keys "
            "'label' (positive, neutral, negative) and 'score' (0-1 confidence). "
            "Only output JSON. Text:\n"
            f"{text}"
        )

        def _call_model() -> str:
            assert self._model is not None
            response = self._model.generate_content(prompt)

            if getattr(response, "text", None):
                return response.text

            if response.candidates:
                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        if getattr(part, "text", None):
                            return part.text

            raise RuntimeError("Gemini response did not contain text.")

        raw = await run_in_threadpool(_call_model)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemini response was not valid JSON.") from exc

        return {
            "label": data.get("label", "neutral"),
            "score": float(data.get("score", 0.0)),
        }
