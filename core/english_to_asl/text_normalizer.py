import re

CONTRACTIONS = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "how's": "how is",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
}

NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}

IRREGULAR_VERBS = {
    "went": "go",
    "gone": "go",
    "ate": "eat",
    "eaten": "eat",
    "came": "come",
    "done": "do",
    "did": "do",
    "said": "say",
    "saw": "see",
    "forgot": "forget",
}


def _expand_contractions(text: str) -> str:
    for source, target in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    return text


def _normalize_token(token: str) -> str:
    if token in NUMBER_WORDS:
        return NUMBER_WORDS[token]
    if token in IRREGULAR_VERBS:
        return IRREGULAR_VERBS[token]
    if len(token) > 4 and token.endswith("ing"):
        base = token[:-3]
        if len(base) > 1 and base[-1] == base[-2]:
            base = base[:-1]
        if base.endswith("y"):
            return base
        return base
    if len(token) > 3 and token.endswith("ied"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("ed"):
        if token.endswith("eed"):
            return token
        base = token[:-2]
        if base.endswith("i"):
            return base[:-1] + "y"
        if base.endswith("e"):
            return base
        return base
    return token


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower().strip()
    text = _expand_contractions(text)
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    normalized_words = [_normalize_token(w) for w in words]
    return " ".join(normalized_words)
