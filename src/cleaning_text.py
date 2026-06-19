import re
import string
import unicodedata
from collections.abc import Iterable, Iterator
from langdetect import DetectorFactory, LangDetectException, detect

SEED = 239
DetectorFactory.seed = SEED

def word_count(text: str) -> int:
    return len(re.compile(r"\b\w+\b", flags=re.UNICODE).findall(text))

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)              # Стандартизация символов
    text = re.compile(r"[^a-zA-Zа-яА-ЯёЁ0-9\s\.,!?\-:;()\"'@#$%&*+=/]").sub(" ", text)
    text = re.compile(r"<[^>\n]{1,500}>").sub(" ", text)    # Удаление остатков разметки
    text = re.compile(r"[ \t\r\f\v]+").sub(" ", text)       # Нормализация пробелов
    text = "\n".join(part.strip() for part in text.splitlines()) 
    text = re.compile(r"\n{3,}").sub("\n\n", text)               # Нормализация перевода строк
    return text.strip()

def divide(text: str, max_words: int) -> list[str]:
    words = text.split()
    if max_words <= 0 or len(words) <= max_words:
        return [text]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

def text_process(
    records: Iterable[dict],
    languages: Iterable[str] = ("en","ru"),
    max_words: int = 700,
    min_words: int = 50
) -> Iterator[dict]:
    allowlist = set(languages)
    for rcrd in records:
        text = normalize(rcrd.get("text", ""))
        if not text:
            continue
        if (word_count(text) < min_words):
            continue

        for i, chunk in enumerate(divide(text, max_words)):
            chunk = normalize(chunk)
            if (word_count(chunk) >= min_words):
                result = dict(rcrd)
                result["text"] = chunk
                result["chunk_id"] = i
                yield result