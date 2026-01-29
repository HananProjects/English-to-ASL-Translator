WH_WORDS = {"who", "what", "where", "when", "why", "how"}
AUX_VERBS = {"is", "are", "am", "was", "were", "do", "does", "did", "will", "would"}
ARTICLES = {"a", "an", "the", "to", "of"}
TIME_WORDS = {"today", "tomorrow", "yesterday", "now", "later"}


def map_grammar(text: str) -> list[str]:
    """
    Convert normalized English text into ASL-style word order.
    """
    words = text.split()

    # Remove articles and auxiliary verbs
    words = [
        w for w in words
        if w not in ARTICLES and w not in AUX_VERBS
    ]

    # Extract time words (ASL puts time first)
    time_words = [w for w in words if w in TIME_WORDS]
    words = [w for w in words if w not in TIME_WORDS]

    # Handle WH-questions (WH word goes last)
    wh_words = [w for w in words if w in WH_WORDS]
    words = [w for w in words if w not in WH_WORDS]

    # Reassemble sentence
    asl_order = []
    asl_order.extend(time_words)
    asl_order.extend(words)
    asl_order.extend(wh_words)

    return asl_order