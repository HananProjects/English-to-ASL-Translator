import time
from core.types import TranslationResult, ReverseTranslationResult
from core.config import MIN_STT_CONFIDENCE
from typing import Optional
from core.confidence import compute_confidence
from core.asl_to_english.pipeline import parse_asl_tokens, asl_to_english_text
from core.english_to_asl.dictionary.asl_signs import ASL_SIGNS

# These imports WILL FAIL until partners implement them — that is OK
_stt_backend = None

def _get_stt_backend():
    global _stt_backend
    if _stt_backend is None:
        from core.english_to_asl.stt_vosk import VoskSTT
        _stt_backend = VoskSTT()
    return _stt_backend
from core.english_to_asl.text_normalizer import normalize_text
from core.english_to_asl.grammar_mapper import map_grammar
from core.english_to_asl.asl_tokenizer import tokenize_asl

def english_to_asl(audio: Optional[bytes] = None,
                   text: Optional[str] = None):
    """
    Main English → ASL pipeline entry point.
    Accepts either raw audio OR text.
    Returns ASL tokens + confidence + latency.
    """

    start_time = time.time()

    if audio is not None:
        try:
            stt = _get_stt_backend()
        except Exception:
            return TranslationResult(
                asl_tokens=[],
                confidence=0.0,
                latency_ms=elapsed_ms(start_time),
                error="STT_NOT_AVAILABLE",
                source_text=""
            )

    # Step 1: Speech to Text (if audio)
    if audio is not None:
        stt = _get_stt_backend()
        text, stt_conf = stt.speech_to_text(audio)
        if stt_conf < MIN_STT_CONFIDENCE:
            return TranslationResult(
                asl_tokens=[],
                confidence=stt_conf,
                latency_ms=elapsed_ms(start_time),
                error="LOW_STT_CONFIDENCE",
                source_text=text or ""
            )

    # Step 2: Normalize text
    normalized_text = normalize_text(text)
    input_token_count = len(normalized_text.split())

    # Step 3: Map grammar
    grammar_tokens = map_grammar(normalized_text)

    # Step 4: ASL tokenization
    asl_tokens = tokenize_asl(grammar_tokens)

    confidence = compute_confidence(
    stt_confidence=stt_conf if audio is not None else None,
    input_tokens=input_token_count,
    output_tokens=len(asl_tokens),
    )

    latency = elapsed_ms(start_time)

    return TranslationResult(
        asl_tokens=asl_tokens,
        confidence=confidence,
        latency_ms=latency,
        source_text=text or ""
    )


def asl_to_english(tokens: Optional[list[str]] = None,
                   text: Optional[str] = None) -> ReverseTranslationResult:
    """
    Reverse pipeline entry point:
    ASL gloss tokens -> English sentence.
    """
    start_time = time.time()

    source_tokens = [t.upper() for t in tokens] if tokens else []
    if text is not None:
        source_tokens = parse_asl_tokens(text)

    if not source_tokens:
        return ReverseTranslationResult(
            english_text="",
            confidence=0.0,
            latency_ms=elapsed_ms(start_time),
            error="NO_ASL_INPUT",
            source_tokens=[],
        )

    english_text = asl_to_english_text(source_tokens)
    known_count = sum(1 for token in source_tokens if token in ASL_SIGNS)
    confidence = known_count / len(source_tokens) if source_tokens else 0.0

    return ReverseTranslationResult(
        english_text=english_text,
        confidence=confidence,
        latency_ms=elapsed_ms(start_time),
        source_tokens=source_tokens,
    )


def elapsed_ms(start_time: float) -> int:
    return int((time.time() - start_time) * 1000)
