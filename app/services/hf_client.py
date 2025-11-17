from __future__ import annotations

import httpx


class HuggingFaceService:
    """Calls the Hugging Face inference endpoint for sentiment analysis."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        timeout: float = 15.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._base_url = f"https://router.huggingface.co/hf-inference/models/{model}"

    async def sentiment(self, text: str) -> dict[str, float | str]:
        cleaned = text.strip()
        if not cleaned:
            return {"label": "neutral", "score": 0.0}
        if not self.api_key:
            raise RuntimeError(
                "HF_API_KEY is required to call the Hugging Face Inference API."
            )
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {"inputs": cleaned[:512], "options": {"wait_for_model": True}}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self._base_url, headers=headers, json=payload
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Hugging Face API error: {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError("Unable to reach Hugging Face API") from exc

        # Responses are typically [[{label, score}, ...]]
        if isinstance(data, list) and data and isinstance(data[0], list):
            candidate = max(data[0], key=lambda item: item.get("score", 0))
        elif isinstance(data, list):
            candidate = max(data, key=lambda item: item.get("score", 0))
        else:
            candidate = data

        raw_label = candidate.get("label", "neutral")
        score = float(candidate.get("score", 0.0))
        normalized_label = self._normalize_label(raw_label, score)
        return {"label": normalized_label, "score": score}

    @staticmethod
    def _normalize_label(label: str, score: float) -> str:
        lower = label.lower()
        if "pos" in lower:
            normalized = "positive"
        elif "neg" in lower:
            normalized = "negative"
        elif "neu" in lower or "neutral" in lower:
            normalized = "neutral"
        else:
            normalized = "neutral"

        # For binary models without a neutral class, treat low-confidence outputs as neutral.
        if normalized in {"positive", "negative"} and 0.45 < score < 0.55:
            return "neutral"

        return normalized


