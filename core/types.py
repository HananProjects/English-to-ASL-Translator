from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TranslationResult:
    asl_tokens: List[str]
    confidence: float
    latency_ms: int
    error: Optional[str] = None
    source_text: str = ""
    used_fingerspelling: bool = False
    spelled_words: Optional[List[str]] = None


@dataclass
class ReverseTranslationResult:
    english_text: str
    confidence: float
    latency_ms: int
    error: Optional[str] = None
    source_tokens: Optional[List[str]] = None
