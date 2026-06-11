from __future__ import annotations

import re
from collections import Counter

from rag_po.math_utils import clamp
from rag_po.models import TweetDocument

UNCERTAIN_MARKERS = {
    "unconfirmed", "allegedly", "reportedly", "rumor", "rumour", "claim", "claims",
    "source says", "sources say", "not confirmed", "breaking",
}

STOPWORDS = {
    "about", "after", "again", "also", "amid", "been", "being", "from", "have",
    "into", "more", "near", "news", "official", "officials", "people", "police",
    "says", "said", "that", "their", "there", "this", "those", "tweet", "with",
    "witness", "witnesses",
}

CRASH_OR_HARM_TERMS = {
    "crash", "crashed", "debris", "dead", "death", "died", "killed", "casualties",
    "hostage", "shooting", "attack", "attacked", "gunman", "explosion",
}

SAFE_OR_DENIAL_TERMS = {
    "safe", "safely", "landed", "landing", "alive", "survived", "rescued",
    "no casualties", "no deaths", "hoax", "fake",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z'-]{3,}", text.lower())
        if token not in STOPWORDS
    }


def _numbers(text: str) -> set[str]:
    # Keep numeric claims such as casualty counts, times, and flight identifiers' numeric parts.
    return {match.group(0).lstrip("0") or "0" for match in re.finditer(r"\b\d+(?:\.\d+)?\b", text)}


def _contains_any(text: str, terms: set[str]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def document_suspicion_score(doc: TweetDocument) -> float:
    """Score single-document uncertainty using only observable text/community signals."""
    deny_ratio = (doc.deny_count / doc.reaction_count) if doc.reaction_count else 0.0
    lower = doc.text.lower()
    marker_score = min(1.0, sum(1 for marker in UNCERTAIN_MARKERS if marker in lower) / 3.0)
    return clamp(0.70 * deny_ratio + 0.30 * marker_score)


def _number_conflict_score(doc_numbers: set[str], majority_numbers: set[str]) -> float:
    if not majority_numbers or not doc_numbers:
        return 0.0
    missing_majority = len(majority_numbers - doc_numbers) / len(majority_numbers)
    minority_numbers = len(doc_numbers - majority_numbers) / len(doc_numbers)
    return clamp(0.55 * missing_majority + 0.45 * minority_numbers)


def _outcome_conflict_score(doc: TweetDocument, majority_has_harm: bool, majority_has_safe: bool) -> float:
    doc_has_harm = _contains_any(doc.text, CRASH_OR_HARM_TERMS)
    doc_has_safe = _contains_any(doc.text, SAFE_OR_DENIAL_TERMS)
    if majority_has_harm and doc_has_safe and not doc_has_harm:
        return 1.0
    if majority_has_safe and doc_has_harm and not doc_has_safe:
        return 1.0
    return 0.0


def _mean_peer_support(doc_tokens: set[str], all_tokens: list[set[str]]) -> float:
    scores = []
    for other in all_tokens:
        if other is doc_tokens:
            continue
        union = doc_tokens | other
        scores.append(len(doc_tokens & other) / len(union) if union else 0.0)
    return sum(scores) / len(scores) if scores else 1.0


def consistency_filter(
    docs: list[TweetDocument],
    deny_ratio_threshold: float = 0.6,
    suspicion_threshold: float = 0.55,
) -> tuple[list[TweetDocument], dict[str, object]]:
    if len(docs) <= 1:
        return docs, {"removed_count": 0, "removed_ids": [], "suspicion_scores": {}}

    token_sets = [_tokens(doc.text) for doc in docs]
    number_counts = Counter(number for doc in docs for number in _numbers(doc.text))
    majority_numbers = {number for number, count in number_counts.items() if count >= 2}
    majority_has_harm = sum(1 for doc in docs if _contains_any(doc.text, CRASH_OR_HARM_TERMS)) > len(docs) / 2
    majority_has_safe = sum(1 for doc in docs if _contains_any(doc.text, SAFE_OR_DENIAL_TERMS)) > len(docs) / 2

    peer_support = [_mean_peer_support(tokens, token_sets) for tokens in token_sets]
    support_baseline = sorted(peer_support)[len(peer_support) // 2] if peer_support else 1.0

    kept: list[TweetDocument] = []
    removed: list[str] = []
    scores: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    for doc, support in zip(docs, peer_support):
        deny_ratio = (doc.deny_count / doc.reaction_count) if doc.reaction_count else 0.0
        single_doc_score = document_suspicion_score(doc)
        number_conflict = _number_conflict_score(_numbers(doc.text), majority_numbers)
        outcome_conflict = _outcome_conflict_score(doc, majority_has_harm, majority_has_safe)
        low_support = clamp((support_baseline - support) / support_baseline) if support_baseline > 0 else 0.0
        cross_doc_score = clamp(0.42 * number_conflict + 0.38 * outcome_conflict + 0.20 * low_support)
        suspicion = max(single_doc_score, cross_doc_score)
        scores[doc.doc_id] = suspicion
        components[doc.doc_id] = {
            "single_doc": single_doc_score,
            "number_conflict": number_conflict,
            "outcome_conflict": outcome_conflict,
            "low_peer_support": low_support,
            "peer_support": support,
            "cross_doc": cross_doc_score,
        }
        if deny_ratio >= deny_ratio_threshold or suspicion >= suspicion_threshold:
            removed.append(doc.doc_id)
        else:
            kept.append(doc)
    return kept, {
        "removed_count": len(removed),
        "removed_ids": removed,
        "suspicion_scores": scores,
        "components": components,
        "majority_numbers": sorted(majority_numbers),
        "majority_has_harm": majority_has_harm,
        "majority_has_safe": majority_has_safe,
    }
