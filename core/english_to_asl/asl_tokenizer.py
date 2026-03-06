from core.english_to_asl.dictionary.asl_signs import ASL_SIGNS
from core.english_to_asl.dictionary.synonyms import SYNONYM_TO_SIGN


def _lookup_sign(word: str) -> str | None:
    sign = SYNONYM_TO_SIGN.get(word)
    if sign:
        return sign

    upper = word.upper()
    if upper in ASL_SIGNS:
        return upper

    # Singular fallback for simple plurals.
    if len(word) > 3 and word.endswith("s"):
        singular = word[:-1]
        sign = SYNONYM_TO_SIGN.get(singular)
        if sign:
            return sign
        singular_upper = singular.upper()
        if singular_upper in ASL_SIGNS:
            return singular_upper

    return None


def tokenize_asl(words: list[str], unknown_policy: str = "keep") -> list[str]:
    """
    Map normalized words into ASL sign tokens.

    unknown_policy:
      - keep: preserve unknown words as uppercase tokens
      - drop: skip unknown words
    """
    tokens: list[str] = []
    for word in words:
        sign = _lookup_sign(word)
        if sign is not None:
            tokens.append(sign)
            continue
        upper_word = word.upper()
        if unknown_policy == "drop":
            continue
        if unknown_policy == "spell":
            for ch in upper_word:
                if ch.isalpha():
                    tokens.append(ch)
            continue
        tokens.append(upper_word)
    return tokens


def tokenize_asl_with_fallback(words: list[str]) -> tuple[list[str], list[str]]:
    """
    Tokenize with a spelling fallback for words missing from ASL_SIGNS.

    Returns:
      - tokens
      - list of unknown words that were spelled letter-by-letter
    """
    tokens: list[str] = []
    spelled_words: list[str] = []
    for word in words:
        sign = _lookup_sign(word)
        if sign is not None:
            tokens.append(sign)
            continue
        spelled = [ch for ch in word.upper() if ch.isalpha()]
        if spelled:
            tokens.extend(spelled)
            spelled_words.append(word)
        else:
            tokens.append(word.upper())
            spelled_words.append(word)
    return tokens, spelled_words
