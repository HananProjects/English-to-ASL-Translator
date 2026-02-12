from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TranslationResult:
    asl_tokens: List[str]
    confidence: float
    latency_ms: int
    error: Optional[str] = None
    source_text: str = ""
