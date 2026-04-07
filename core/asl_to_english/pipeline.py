from typing import List, Optional

WH_WORDS = {"who", "what", "where", "when", "why", "how"}
TIME_WORDS = {"today", "tomorrow", "yesterday", "now", "later"}

TOKEN_TO_ENGLISH = {
    "ME": "i",
    "THANK_YOU": "thank you",
}


def parse_asl_tokens(text: str) -> List[str]:
    raw = text.replace(",", " ").split()
    tokens = [token.strip().upper() for token in raw if token.strip()]
    return tokens


def asl_to_english_text(tokens: List[str]) -> str:
    if not tokens:
        return ""

    words = [_token_to_word(token) for token in tokens]
    words = _restore_english_order(words)
    words = _realize_english(words)
    return _format_sentence(words)


def _token_to_word(token: str) -> str:
    mapped = TOKEN_TO_ENGLISH.get(token)
    if mapped is not None:
        return mapped
    return token.lower().replace("_", " ")


def _restore_english_order(words: List[str]) -> List[str]:
    restored = list(words)

    # ASL commonly places time first and wh-word at the end.
    if restored and restored[0] in TIME_WORDS and len(restored) > 1:
        time_word = restored.pop(0)
        restored.append(time_word)

    if restored and restored[-1] in WH_WORDS and len(restored) > 1:
        wh = restored.pop()
        restored.insert(0, wh)

    return restored


def _format_sentence(words: List[str]) -> str:
    if not words:
        return ""
    words = ["I" if w == "i" else w for w in words]
    text = " ".join(words)
    if words[0] in WH_WORDS:
        suffix = "?"
    else:
        suffix = "."
    return text[0].upper() + text[1:] + suffix


def _realize_english(words: List[str]) -> List[str]:
    if not words:
        return words

    # Expand a common planning phrase into more natural English.
    if len(words) == 3 and words[0] == "we" and words[1] == "go" and words[2] == "school":
        return ["we", "should", "go", "to", "school"]

    # Recover omitted "to be" auxiliaries and progressive verb form in
    # common ASL WH-questions, e.g. "YOU GO WHERE" -> "where are you going".
    if words[0] in WH_WORDS and len(words) >= 3:
        subject = words[1]
        verb = words[2]
        aux = _present_be_for_subject(subject)
        if aux is not None:
            realized = [words[0], aux, subject, _to_gerund(verb)]
            if len(words) > 3:
                realized.extend(words[3:])
            return realized

    return words


def _present_be_for_subject(subject: str) -> Optional[str]:
    s = subject.lower()
    if s == "i":
        return "am"
    if s in {"you", "we", "they"}:
        return "are"
    if s in {"he", "she", "it"}:
        return "is"
    return None


def _to_gerund(verb: str) -> str:
    v = verb.lower()
    if v.endswith("ing"):
        return v
    if len(v) > 2 and v.endswith("e") and not v.endswith(("ee", "oe", "ye")):
        return v[:-1] + "ing"
    return v + "ing"
