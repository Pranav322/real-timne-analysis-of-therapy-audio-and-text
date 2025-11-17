from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf


@dataclass
class ProsodyResult:
    label: str
    score: float
    duration_seconds: float


class ProsodyAnalyzer:
    """Lightweight audio feature extractor for stress proxies."""

    def analyze(self, audio_bytes: bytes) -> ProsodyResult:
        if not audio_bytes:
            return ProsodyResult(label="unknown", score=0.0, duration_seconds=0.0)
        samples, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        duration = float(len(samples) / sample_rate) if sample_rate else 0.0
        if duration == 0:
            return ProsodyResult(label="unknown", score=0.0, duration_seconds=0.0)

        rms = float(np.sqrt(np.mean(np.square(samples))))
        zero_crossings = np.nonzero(np.diff(np.signbit(samples)))[0]
        zcr_rate = float(len(zero_crossings)) / duration

        # Combine normalized features into a 0-1 stress score.
        normalized_rms = min(rms * 12, 1.0)
        normalized_zcr = min(zcr_rate / 300, 1.0)
        stress_score = min((normalized_rms * 0.6) + (normalized_zcr * 0.4), 1.0)

        if stress_score < 0.35:
            label = "calm"
        elif stress_score < 0.65:
            label = "elevated"
        else:
            label = "high"

        return ProsodyResult(label=label, score=stress_score, duration_seconds=duration)



