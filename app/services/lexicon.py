from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class LexiconSentiment:
    """Lightweight offline sentiment fallback powered by VADER."""

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> dict[str, float | str]:
        if not text.strip():
            return {"label": "neutral", "score": 0.0}
        scores = self._analyzer.polarity_scores(text)
        compound = scores.get("compound", 0.0)
        if compound > 0.15:
            label = "positive"
            confidence = min(1.0, 0.5 + compound / 2)
        elif compound < -0.15:
            label = "negative"
            confidence = min(1.0, 0.5 + abs(compound) / 2)
        else:
            label = "neutral"
            confidence = 0.5 - abs(compound) / 4
        return {"label": label, "score": round(confidence, 3)}

