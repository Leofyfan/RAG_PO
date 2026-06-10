from __future__ import annotations

from statistics import mean, pstdev

from rag_po.math_utils import cosine_distance
from rag_po.models import TweetDocument


def semantic_outlier_filter(
    retrieved_docs: list[TweetDocument],
    z_threshold: float = 2.0,
    min_docs: int = 4,
) -> tuple[list[TweetDocument], dict[str, object]]:
    if len(retrieved_docs) < min_docs:
        return retrieved_docs, {"removed_count": 0, "threshold": None, "distances": {}}
    if any(doc.embedding is None for doc in retrieved_docs):
        return retrieved_docs, {"removed_count": 0, "threshold": None, "reason": "missing_embeddings"}

    distances: dict[str, float] = {}
    for i, doc in enumerate(retrieved_docs):
        others = [cosine_distance(doc.embedding, other.embedding) for j, other in enumerate(retrieved_docs) if i != j]
        distances[doc.doc_id] = mean(others) if others else 0.0
    vals = list(distances.values())
    threshold = mean(vals) + z_threshold * pstdev(vals)
    kept = [doc for doc in retrieved_docs if distances[doc.doc_id] <= threshold]
    removed = [doc.doc_id for doc in retrieved_docs if distances[doc.doc_id] > threshold]
    return kept, {"removed_count": len(removed), "removed_ids": removed, "threshold": threshold, "distances": distances}
