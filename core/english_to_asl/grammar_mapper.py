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
GREETING_WORDS = {"hello", "hi", "hey"}


def map_grammar(text: str) -> list[str]:
    """
    Convert normalized English text into ASL-style word order.
    """
    words = text.split()
    if not words:
        return []

    # Common conversational phrase handling with copula drop:
    # "hello how are you" -> "hello how you"
    has_how_are_you = "how" in words and "you" in words and "are" in words
    greeting = next((w for w in words if w in GREETING_WORDS), None)
    if has_how_are_you and greeting is not None:
        return [greeting, "how", "you"]
    if has_how_are_you and words[0] == "how":
        return ["how", "you"]

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
