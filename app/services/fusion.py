from __future__ import annotations

from typing import List

from ..schemas import InsightPayload, SentimentResult, StressResult


class InsightEngine:
    """Combines sentiment + prosody cues into therapist-facing insights."""

    def build_insight(
        self,
        sentiment: SentimentResult | None,
        stress: StressResult | None,
        crisis_triggered: bool = False,
    ) -> InsightPayload:
        sentiment_label = sentiment.label if sentiment else "neutral"
        stress_label = stress.label if stress else "unknown"
        stress_score = stress.score if stress else 0.0

        risk = "low"
        recommendations: List[str] = []

        if crisis_triggered:
            risk = "critical"
            recommendations.append(
                "Immediate risk detected. Pause session, follow safety protocol, and escalate."
            )
        elif sentiment_label == "negative" and stress_score > 0.65:
            risk = "high"
            recommendations.append(
                "Pause and validate emotions; assess for crisis indicators."
            )
        elif sentiment_label == "negative" or stress_score > 0.5:
            risk = "moderate"
            recommendations.append("Explore triggers and coping strategies.")

        if sentiment_label == "positive" and stress_score < 0.35:
            recommendations.append("Reinforce recent progress.")

        engagement = self._derive_engagement(sentiment_label, stress_score)

        if not recommendations:
            recommendations.append("Continue reflective listening.")

        return InsightPayload(
            engagement=engagement,
            stress=stress_label,
            risk=risk,
            recommendations=recommendations,
        )

    @staticmethod
    def _derive_engagement(sentiment_label: str, stress_score: float) -> str:
        if sentiment_label == "positive" and stress_score < 0.4:
            return "high"
        if stress_score > 0.75:
            return "low"
        return "medium"



