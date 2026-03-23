from core.engine import english_to_asl


def test_contraction_wh_question():
    r = english_to_asl(text="Where's your brother?")
    assert r.asl_tokens == ["YOUR", "BROTHER", "WHERE"]


def test_negation_question():
    r = english_to_asl(text="Why are you not working?")
    assert r.asl_tokens == ["YOU", "WORK", "NO", "WHY"]


def test_time_marker_fronting():
    r = english_to_asl(text="I am eating now")
    assert r.asl_tokens == ["NOW", "ME", "EAT"]


def test_synonym_mapping():
    r = english_to_asl(text="Thanks")
    assert r.asl_tokens == ["THANK_YOU"]


def test_unknown_token_preserved():
    r = english_to_asl(text="I go home")
    assert r.asl_tokens == ["ME", "GO", "HOME"]


def test_greeting_how_are_you_order():
    r = english_to_asl(text="hello how are you")
    assert r.asl_tokens == ["HELLO", "HOW", "YOU"]


def test_how_are_you_order():
    r = english_to_asl(text="how are you")
    assert r.asl_tokens == ["HOW", "YOU"]


def test_greeting_how_are_you_preserves_extra_words():
    r = english_to_asl(text="hello how are you doing my friend")
    assert r.asl_tokens == ["HELLO", "YOU", "MY", "FRIEND", "HOW"]
