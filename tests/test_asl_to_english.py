from core.engine import asl_to_english


def test_wh_question_reverse():
    r = asl_to_english(tokens=["YOU", "GO", "WHERE"])
    assert r.english_text == "Where you go?"
    assert r.error is None


def test_time_word_reverse():
    r = asl_to_english(tokens=["TOMORROW", "I", "GO"])
    assert r.english_text == "I go tomorrow."


def test_text_input_reverse():
    r = asl_to_english(text="YOU GO WHERE")
    assert r.english_text == "Where you go?"
