from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from rag_po.defense.social_rerank import event_average_sentiment, social_credibility_score
from rag_po.math_utils import clamp
from rag_po.models import TweetDocument

HARM_TERMS = {
    "affected", "casualties", "dead", "death", "died", "hostage", "injured",
    "killed", "shooting", "victims",
}

SAFE_TERMS = {"alive", "fake", "hoax", "landed", "rescued", "safe", "survived"}


def extract_claim(doc: TweetDocument) -> dict[str, Any]:
    """Extract lightweight claim slots from one document without using hidden labels."""
    lower = doc.text.lower()
    numbers = sorted({match.group(0).lstrip("0") or "0" for match in re.finditer(r"\b\d+(?:\.\d+)?\b", lower)})
    has_harm = any(term in lower for term in HARM_TERMS)
    has_safe = any(term in lower for term in SAFE_TERMS)
    if has_harm and not has_safe:
        outcome = "harm"
    elif has_safe and not has_harm:
        outcome = "safe"
    elif has_harm and has_safe:
        outcome = "mixed"
    else:
        outcome = "neutral"
    authoritative = any(token in lower for token in ["official", "police", "statement", "confirmed", "reuters"])
    uncertain = any(token in lower for token in ["allegedly", "claim", "claims", "rumor", "rumour", "unconfirmed"])
    return {
        "numbers": numbers,
        "outcome": outcome,
        "authoritative": authoritative,
        "uncertain": uncertain,
    }


def _weighted_votes(records: dict[str, dict[str, Any]], weights: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    number_votes: dict[str, float] = {}
    outcome_votes: dict[str, float] = {}
    for doc_id, claim in records.items():
        weight = weights.get(doc_id, 0.0)
        for number in claim["numbers"]:
            number_votes[number] = number_votes.get(number, 0.0) + weight
        outcome = claim["outcome"]
        if outcome != "neutral":
            outcome_votes[outcome] = outcome_votes.get(outcome, 0.0) + weight
    return number_votes, outcome_votes


def _support_score(claim: dict[str, Any], number_winners: set[str], outcome_winner: str | None) -> float:
    parts: list[float] = []
    if number_winners:
        parts.append(1.0 if set(claim["numbers"]) & number_winners else 0.0)
    if outcome_winner:
        parts.append(1.0 if claim["outcome"] == outcome_winner else 0.0)
    if claim["authoritative"]:
        parts.append(0.65)
    if claim["uncertain"]:
        parts.append(0.25)
    return sum(parts) / len(parts) if parts else 0.5


def trust_weighted_claim_rank(
    docs: list[TweetDocument],
    semantic_scores: dict[str, float],
    cluster_sizes: dict[str, int] | None = None,
) -> tuple[list[TweetDocument], dict[str, object]]:
    """Rank isolated document claims by observable trust and weighted support.

    This approximates the plan's isolate-extract-aggregate step without leaking
    labels or attack markers into the defense.
    """
    if not docs:
        return [], {"scores": {}, "number_votes": {}, "outcome_votes": {}}
    cluster_sizes = cluster_sizes or {}
    avg_sentiment = event_average_sentiment(docs)
    claims = {doc.doc_id: extract_claim(doc) for doc in docs}
    trust_scores = {doc.doc_id: social_credibility_score(doc, avg_sentiment) for doc in docs}
    weights = {
        doc.doc_id: trust_scores[doc.doc_id] / max(1, int(cluster_sizes.get(doc.doc_id, 1)))
        for doc in docs
    }
    number_votes, outcome_votes = _weighted_votes(claims, weights)
    max_number_vote = max(number_votes.values(), default=0.0)
    number_winners = {number for number, vote in number_votes.items() if vote == max_number_vote and vote > 0.0}
    outcome_winner = max(outcome_votes, key=outcome_votes.get) if outcome_votes else None

    enriched: list[TweetDocument] = []
    scores: dict[str, dict[str, float]] = {}
    for doc in docs:
        semantic = clamp(float(semantic_scores.get(doc.doc_id, 0.0)))
        trust = trust_scores[doc.doc_id]
        support = _support_score(claims[doc.doc_id], number_winners, outcome_winner)
        final = clamp(0.45 * semantic + 0.40 * trust + 0.15 * support)
        metadata = dict(doc.metadata or {})
        defense_meta = dict(metadata.get("defense") or {})
        defense_meta.update(
            {
                "trust": trust,
                "claim_support": support,
                "claim_weight": weights[doc.doc_id],
                "cluster_size": max(1, int(cluster_sizes.get(doc.doc_id, 1))),
                "weighted_score": final,
            }
        )
        metadata["defense"] = defense_meta
        enriched.append(replace(doc, metadata=metadata))
        scores[doc.doc_id] = {
            "semantic": semantic,
            "trust": trust,
            "claim_support": support,
            "claim_weight": weights[doc.doc_id],
            "final": final,
        }

    ranked = sorted(enriched, key=lambda doc: scores[doc.doc_id]["final"], reverse=True)
    return ranked, {
        "scores": scores,
        "claims": claims,
        "number_votes": number_votes,
        "outcome_votes": outcome_votes,
        "number_winners": sorted(number_winners),
        "outcome_winner": outcome_winner,
    }
