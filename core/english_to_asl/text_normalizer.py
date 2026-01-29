import re

VERB_NORMALIZATION = {
    "going": "go",
    "eating": "eat",
    "working": "work",
    "coming": "come",
    "doing": "do",
    "saying": "say"
}

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)

    words = text.split()
    normalized_words = [
        VERB_NORMALIZATION.get(w, w)
        for w in words
    ]

    return " ".join(normalized_words)