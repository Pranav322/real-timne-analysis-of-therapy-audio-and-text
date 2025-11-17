from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass
class CrisisSignal:
    triggered: bool
    keyword: str | None = None


class CrisisDetector:
    """Rule-based detector for severe risk phrases."""

    DEFAULT_PATTERNS = [
        r"\b(kill myself|kill me|suicide|end my life)\b",
        r"\bi want to die\b",
        r"\b(can't|cannot) go on\b",
        r"\bi won't make it\b",
        r"\bno reason to live\b",
        r"\bhurt myself\b",
        r"\bplan to (die|kill)\b",
        r"\b(cut|cutting) (my|the) (wrist|wrists|arm|arms|hand|hands)\b",
        r"\bcut myself\b",
        r"\b(killed|kill) (my|his|her|their) (brother|sister|mother|father|family|friend)\b",
        r"\b(killed|kill) (himself|herself|themselves)\b",
        r"\b(committed|commit) suicide\b",
        r"\bmurder(ed)? (my|his|her|their)\b",
    ]

    def __init__(self, patterns: Iterable[str] | None = None) -> None:
        combined = patterns or self.DEFAULT_PATTERNS
        self._regexes = [re.compile(pattern, re.IGNORECASE) for pattern in combined]

    def detect(self, text: str) -> CrisisSignal:
        for pattern in self._regexes:
            match = pattern.search(text)
            if match:
                return CrisisSignal(triggered=True, keyword=match.group(0))
        return CrisisSignal(triggered=False, keyword=None)

