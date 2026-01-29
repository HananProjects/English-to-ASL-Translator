def map_grammar(text: str) -> list[str]:
    words = text.split()

    if "where" in words:
        words.remove("where")
        words.append("where")

    return words