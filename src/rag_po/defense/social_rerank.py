from __future__ import annotations

import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from rag_po.math_utils import clamp
from rag_po.models import TweetDocument
from rag_po.text_utils import contains_url, count_extreme_punctuation, simple_sentiment

REFERENCE_DATE = datetime(2016, 1, 1, tzinfo=timezone.utc)


def _log_norm(value: float, cap: float) -> float:
    return clamp(math.log1p(max(0.0, value)) / math.log1p(cap))


def _parse_twitter_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def account_age_years(created_at: str, reference: datetime = REFERENCE_DATE) -> float:
    dt = _parse_twitter_date(created_at)
    if dt is None:
        return 1.0
    return max(0.0, (reference - dt).days / 365.25)


def user_credibility_score(doc: TweetDocument) -> float:
    user = doc.user or {}
    followers = float(user.get("followers_count") or 0)
    friends = float(user.get("friends_count") or 0)
    listed = float(user.get("listed_count") or 0)
    statuses = float(user.get("statuses_count") or 0)
    verified = 1.0 if user.get("verified") else 0.0
    default_profile = 1.0 if user.get("default_profile") else 0.0
    age = max(0.1, account_age_years(str(user.get("created_at") or "")))
    ratio = followers / (friends + 1.0)
    activity_per_day = statuses / max(1.0, age * 365.25)
    activity_score = 1.0 - clamp(max(0.0, activity_per_day - 80.0) / 200.0)
    return clamp(
        0.25 * _log_norm(followers, 1_000_000)
        + 0.16 * _log_norm(ratio, 100)
        + 0.20 * verified
        + 0.14 * _log_norm(age, 8)
        + 0.12 * _log_norm(listed, 10_000)
        + 0.08 * activity_score
        + 0.05 * (1.0 - default_profile)
    )


def content_credibility_score(doc: TweetDocument) -> float:
    text = doc.text.lower()
    authoritative = any(token in text for token in ["official", "police", "statement", "confirmed", "reuters", "bbc", "according to"])
    uncertainty = any(token in text for token in ["unconfirmed", "allegedly", "rumor", "rumour", "claims"])
    punct_penalty = clamp(count_extreme_punctuation(doc.text) / 8.0)
    url_bonus = 0.08 if contains_url(doc.text) else 0.0
    score = 0.55 + (0.18 if authoritative else 0.0) - (0.18 if uncertainty else 0.0) - 0.14 * punct_penalty + url_bonus
    return clamp(score)


def propagation_score(doc: TweetDocument) -> float:
    deny_ratio = doc.deny_count / doc.reaction_count if doc.reaction_count else 0.0
    engagement = _log_norm(doc.retweet_count + doc.favorite_count + doc.reaction_count, 100_000)
    return clamp(0.65 * (1.0 - deny_ratio) + 0.35 * engagement)


def emotion_normality_score(doc: TweetDocument, event_avg_sentiment: float = 0.0) -> float:
    return clamp(1.0 - abs(simple_sentiment(doc.text) - event_avg_sentiment))


def social_credibility_score(doc: TweetDocument, event_avg_sentiment: float = 0.0) -> float:
    return clamp(
        0.42 * user_credibility_score(doc)
        + 0.18 * emotion_normality_score(doc, event_avg_sentiment)
        + 0.20 * propagation_score(doc)
        + 0.20 * content_credibility_score(doc)
    )


def event_average_sentiment(docs: list[TweetDocument]) -> float:
    if not docs:
        return 0.0
    return sum(simple_sentiment(doc.text) for doc in docs) / len(docs)


def social_rerank(
    docs: list[TweetDocument],
    semantic_scores: dict[str, float],
    alpha: float = 0.65,
) -> list[TweetDocument]:
    avg_sentiment = event_average_sentiment(docs)

    def final_score(doc: TweetDocument) -> float:
        sem = semantic_scores.get(doc.doc_id, 0.0)
        cred = social_credibility_score(doc, avg_sentiment)
        return alpha * sem + (1.0 - alpha) * cred

    return sorted(docs, key=final_score, reverse=True)


def social_rerank_with_audit(
    docs: list[TweetDocument],
    semantic_scores: dict[str, float],
    alpha: float = 0.65,
) -> tuple[list[TweetDocument], dict[str, object]]:
    avg_sentiment = event_average_sentiment(docs)
    scores = {
        doc.doc_id: {
            "semantic": semantic_scores.get(doc.doc_id, 0.0),
            "credibility": social_credibility_score(doc, avg_sentiment),
            "final": alpha * semantic_scores.get(doc.doc_id, 0.0)
            + (1.0 - alpha) * social_credibility_score(doc, avg_sentiment),
        }
        for doc in docs
    }
    ranked = sorted(docs, key=lambda doc: scores[doc.doc_id]["final"], reverse=True)
    return ranked, {"scores": scores, "alpha": alpha}
