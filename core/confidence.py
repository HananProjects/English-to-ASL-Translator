from typing import Optional


def compute_confidence(
    *,
    stt_confidence: Optional[float],
    input_tokens: int,
    output_tokens: int
) -> float:
    """
    Compute composite translation confidence.

    - stt_confidence: confidence reported by STT (0–1), or None for text input
    - input_tokens: number of tokens before grammar
    - output_tokens: number of ASL tokens after grammar
    """

    # STT contribution
    if stt_confidence is None:
        stt_score = 1.0
    else:
        stt_score = max(0.0, min(stt_confidence, 1.0))

    # Grammar coverage
    if input_tokens == 0:
        grammar_score = 0.0
    else:
        grammar_score = min(output_tokens / input_tokens, 1.0)

    # Output validity
    output_score = 1.0 if output_tokens > 0 else 0.0

    # Weighted sum
    confidence = (
        0.6 * stt_score +
        0.3 * grammar_score +
        0.1 * output_score
    )

    # Hard penalty: no ASL output means low confidence
    if output_tokens == 0:
        confidence = min(confidence, 0.4)

    return round(confidence, 3)
