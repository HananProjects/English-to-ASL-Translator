from core.confidence import compute_confidence


def test_text_only_confidence():
    c = compute_confidence(
        stt_confidence=None,
        input_tokens=4,
        output_tokens=3
    )
    assert 0.8 <= c <= 1.0


def test_low_grammar_coverage():
    c = compute_confidence(
        stt_confidence=0.9,
        input_tokens=5,
        output_tokens=1
    )
    assert c < 0.8


def test_empty_output_confidence():
    c = compute_confidence(
        stt_confidence=0.9,
        input_tokens=3,
        output_tokens=0
    )
    assert c < 0.5