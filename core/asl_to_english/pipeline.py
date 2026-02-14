from typing import List

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
    text = " ".join(words)
    if words[0] in WH_WORDS:
        suffix = "?"
    else:
        suffix = "."
    return text[0].upper() + text[1:] + suffix
