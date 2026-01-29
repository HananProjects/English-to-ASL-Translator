def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("?", "")
    text = text.replace("are", "")
    text = text.replace("is", "")
    return text.strip()