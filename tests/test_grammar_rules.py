from core.engine import english_to_asl


def test_wh_question():
    r = english_to_asl(text="Where are you going?")
    assert r.asl_tokens == ["YOU", "GO", "WHERE"]


def test_time_first():
    r = english_to_asl(text="I will go tomorrow")
    assert r.asl_tokens == ["TOMORROW", "ME", "GO"]


def test_statement():
    r = english_to_asl(text="I am going home")
    assert r.asl_tokens == ["ME", "GO", "HOME"]
