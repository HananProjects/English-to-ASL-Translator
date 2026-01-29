import time
from core.types import TranslationResult
from core.config import MIN_STT_CONFIDENCE

# These imports WILL FAIL until partners implement them — that is OK
from core.english_to_asl.stt import speech_to_text
from core.english_to_asl.text_normalizer import normalize_text
from core.english_to_asl.grammar_mapper import map_grammar
from core.english_to_asl.asl_tokenizer import tokenize_asl


def english_to_asl(audio: bytes | None = None,
                   text: str | None = None) -> TranslationResult:
    """
    Main English → ASL pipeline entry point.
    Accepts either raw audio OR text.
    Returns ASL tokens + confidence + latency.
    """

    start_time = time.time()

    if audio is None and text is None:
        return TranslationResult(
            asl_tokens=[],
            confidence=0.0,
            latency_ms=0,
            error="NO_INPUT"
        )

    # Step 1: Speech to Text (if audio)
    if audio is not None:
        text, stt_conf = speech_to_text(audio)

        if stt_conf < MIN_STT_CONFIDENCE:
            return TranslationResult(
                asl_tokens=[],
                confidence=stt_conf,
                latency_ms=elapsed_ms(start_time),
                error="LOW_STT_CONFIDENCE"
            )

    # Step 2: Normalize text
    normalized_text = normalize_text(text)

    # Step 3: Map grammar
    grammar_tokens = map_grammar(normalized_text)

    # Step 4: ASL tokenization
    asl_tokens = tokenize_asl(grammar_tokens)

    latency = elapsed_ms(start_time)

    return TranslationResult(
        asl_tokens=asl_tokens,
        confidence=1.0,  # placeholder until confidence logic is added
        latency_ms=latency
    )


def elapsed_ms(start_time: float) -> int:
    return int((time.time() - start_time) * 1000)