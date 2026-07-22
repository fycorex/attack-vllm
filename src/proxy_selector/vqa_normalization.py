"""Subset of the official VQAv2 answer normalization rules."""

from __future__ import annotations

import re

_ARTICLES = {"a", "an", "the"}
_NUMBERS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_CONTRACTIONS = {"cant": "can't", "wont": "won't", "dont": "don't", "isnt": "isn't", "arent": "aren't", "couldnt": "couldn't", "wouldnt": "wouldn't"}


def normalize_answer(answer: str) -> str:
    """Normalize a short answer consistently for screening and scoring."""
    value = answer.lower().strip()
    value = re.sub(r"(?<!\d)[,](?!\d)", "", value)
    value = re.sub(r"[;\/\[\]\"{}()=+\\_><@`?,!]+", " ", value)
    value = re.sub(r"(?<!\d)\.(?!\d)", " ", value)
    words = []
    for word in value.split():
        word = _CONTRACTIONS.get(word, word)
        word = _NUMBERS.get(word, word)
        if word not in _ARTICLES:
            words.append(word)
    return " ".join(words)
