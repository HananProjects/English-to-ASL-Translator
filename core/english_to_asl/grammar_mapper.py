WH_WORDS = {"who", "what", "where", "when", "why", "how"}
AUX_VERBS = {
    "is",
    "are",
    "am",
    "was",
    "were",
    "do",
    "does",
    "did",
    "will",
    "would",
    "can",
    "could",
    "have",
    "has",
    "had",
}
ARTICLES = {"a", "an", "the", "to", "of", "for"}
TIME_WORDS = {"today", "tomorrow", "yesterday", "now", "later"}
NEGATION_WORDS = {"not", "never", "no"}


def map_grammar(text: str) -> list[str]:
    """
    Convert normalized English text into ASL-style word order.
    """
    words = text.split()
    if not words:
        return []

    # ASL commonly fronts time markers.
    time_words = [w for w in words if w in TIME_WORDS]

    # Model sentence negation as an explicit "NO" token.
    has_negation = any(w in NEGATION_WORDS for w in words)

    core_words = [
        w
        for w in words
        if w not in TIME_WORDS
        and w not in NEGATION_WORDS
        and w not in ARTICLES
        and w not in AUX_VERBS
    ]

    # In many ASL question structures, WH words are sentence-final.
    wh_words = [w for w in core_words if w in WH_WORDS]
    non_wh_words = [w for w in core_words if w not in WH_WORDS]

    asl_order = [*time_words, *non_wh_words]
    if has_negation:
        asl_order.append("no")
    asl_order.extend(wh_words)

    return asl_order
