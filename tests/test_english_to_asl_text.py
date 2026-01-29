from core.engine import english_to_asl

def test_where_question():
    r = english_to_asl(text="Where are you going")
    assert r.asl_tokens == ["YOU", "GO", "WHERE"]

    