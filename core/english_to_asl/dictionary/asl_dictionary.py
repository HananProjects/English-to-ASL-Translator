class ASLDictionary:
    def __init__(self):
        # English token -> ASL sign ID
        self.lexicon = {
            "hello": "HELLO",
            "hi": "HELLO",
            "hey": "HELLO",

            "thank": "THANK_YOU",
            "thanks": "THANK_YOU",

            "you": "YOU",
            "me": "ME",
            "i": "ME",

            "yes": "YES",
            "no": "NO",
        }

    def lookup(self, token: str) -> str | None:
        """
        Returns ASL sign ID or None if unknown
        """
        return self.lexicon.get(token)
