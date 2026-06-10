from __future__ import annotations

import html
import re

URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
SPACE_RE = re.compile(r"\s+")


def clean_tweet_text(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\u2026", "...")
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def compact_for_prompt(text: str, max_chars: int = 500) -> str:
    text = clean_tweet_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def contains_url(text: str) -> bool:
    return bool(URL_RE.search(text or ""))


def count_extreme_punctuation(text: str) -> int:
    return sum(1 for c in text if c in "!?")


def simple_sentiment(text: str) -> float:
    """Small lexicon sentiment in [-1, 1], avoiding external model downloads."""
    positive = {
        "safe", "confirmed", "official", "help", "rescued", "peace", "support",
        "solidarity", "survived", "calm", "good", "relief",
    }
    negative = {
        "kill", "killed", "dead", "death", "attack", "terror", "terrorist", "hostage",
        "shooting", "bomb", "crash", "panic", "fear", "hate", "horror", "wounded",
        "missing", "explosion", "gunman", "danger", "hoax", "fake",
    }
    tokens = re.findall(r"[a-zA-Z']+", (text or "").lower())
    if not tokens:
        return 0.0
    score = sum(1 for t in tokens if t in positive) - sum(1 for t in tokens if t in negative)
    return max(-1.0, min(1.0, score / max(4, len(tokens) ** 0.5)))
