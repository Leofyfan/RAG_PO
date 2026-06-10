from __future__ import annotations

from rag_po.attack.random_inject import injection_count
from rag_po.math_utils import cosine_similarity
from rag_po.models import TweetDocument


def build_semantic_poisoned_corpus(
    clean_docs: list[TweetDocument],
    rumour_pool: list[TweetDocument],
    query_embedding: list[float],
    ratio: float,
) -> list[TweetDocument]:
    n = injection_count(len(clean_docs), ratio, len(rumour_pool))
    ranked = sorted(
        rumour_pool,
        key=lambda doc: cosine_similarity(query_embedding, doc.embedding),
        reverse=True,
    )
    return list(clean_docs) + [doc.with_attack("semantic") for doc in ranked[:n]]
