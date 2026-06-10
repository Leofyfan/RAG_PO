from __future__ import annotations

from collections import Counter

from rag_po.models import TweetDocument

UNCERTAIN_MARKERS = {
    "unconfirmed", "allegedly", "reportedly", "rumor", "rumour", "claim", "claims",
    "source says", "sources say", "not confirmed", "breaking",
}


def document_suspicion_score(doc: TweetDocument) -> float:
    deny_ratio = (doc.deny_count / doc.reaction_count) if doc.reaction_count else 0.0
    lower = doc.text.lower()
    marker_score = min(1.0, sum(1 for marker in UNCERTAIN_MARKERS if marker in lower) / 3.0)
    rumour_bonus = 0.2 if doc.attack else 0.0
    return min(1.0, 0.65 * deny_ratio + 0.25 * marker_score + rumour_bonus)


def consistency_filter(
    docs: list[TweetDocument],
    deny_ratio_threshold: float = 0.6,
    suspicion_threshold: float = 0.82,
) -> tuple[list[TweetDocument], dict[str, object]]:
    kept: list[TweetDocument] = []
    removed: list[str] = []
    scores: dict[str, float] = {}
    for doc in docs:
        deny_ratio = (doc.deny_count / doc.reaction_count) if doc.reaction_count else 0.0
        suspicion = document_suspicion_score(doc)
        scores[doc.doc_id] = suspicion
        if deny_ratio >= deny_ratio_threshold or suspicion >= suspicion_threshold:
            removed.append(doc.doc_id)
        else:
            kept.append(doc)
    return kept, {"removed_count": len(removed), "removed_ids": removed, "suspicion_scores": scores}
