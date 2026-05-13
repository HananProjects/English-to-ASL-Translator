# Sample file with intentional issues for testing the AI code reviewer.
# DO NOT use in production.

import sqlite3
import requests

# Hardcoded secret (security: critical)
API_KEY = "sk-ant-api03-FAKE1234567890abcdefghijklmnop"
DB_PATH = "asl.db"


def get_sign(word):
    # SQL injection vulnerability (security: high)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM signs WHERE word = '" + word + "'")
    result = cursor.fetchone()
    conn.close()
    return result


def translate_sentence(sentence):
    words = sentence.split(" ")
    signs = []
    # N+1 pattern: separate DB connection per word (performance: high)
    for w in words:
        signs.append(get_sign(w))
    return signs


def translate_all_sentences(sentences):
    # O(n²) — calls translate_sentence which already loops (performance: medium)
    x = []
    for s in sentences:
        for w in s.split(" "):
            x.append(get_sign(w))
    return x


def callApi(w, k):
    # No input validation, bare except (style: block)
    try:
        r = requests.get("https://api.example.com/asl?word=" + w + "&key=" + k)
        return r.json()
    except:
        return None


def p(data):
    # Meaningless name, no type hints (style: nit)
    for i in data:
        if i != None:
            print(i)


def translate_sentence_2(sentence):
    # Duplicated logic from translate_sentence (style: block)
    words = sentence.split(" ")
    signs = []
    for w in words:
        signs.append(get_sign(w))
    return signs
