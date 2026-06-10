from __future__ import annotations

from rag_po.defense.consistency import consistency_filter
from rag_po.defense.outlier_detect import semantic_outlier_filter
from rag_po.defense.social_rerank import social_rerank_with_audit
from rag_po.models import DEFENSE_SPECS, TweetDocument
from rag_po.rag.vectordb import InMemoryVectorStore


def retrieve_with_defense(
    corpus: list[TweetDocument],
    query_embedding: list[float],
    defense: str = "D0",
    retrieve_k: int = 10,
    final_k: int = 5,
) -> tuple[list[TweetDocument], dict[str, object], dict[str, float]]:
    if defense not in DEFENSE_SPECS:
        raise ValueError(f"Unknown defense: {defense}")
    spec = DEFENSE_SPECS[defense]
    store = InMemoryVectorStore(corpus)
    initial_results = store.search(query_embedding, top_k=retrieve_k)
    docs = [item.doc for item in initial_results]
    semantic_scores = {item.doc.doc_id: item.score for item in initial_results}
    audit: dict[str, object] = {
        "initial_ids": [doc.doc_id for doc in docs],
        "initial_scores": semantic_scores,
        "defense": defense,
    }

    if spec.outlier:
        docs, outlier_audit = semantic_outlier_filter(docs)
        audit["outlier"] = outlier_audit
    if spec.consistency:
        docs, consistency_audit = consistency_filter(docs)
        audit["consistency"] = consistency_audit
    if spec.social:
        docs, social_audit = social_rerank_with_audit(docs, semantic_scores)
        audit["social"] = social_audit

    final_docs = docs[:final_k]
    audit["final_ids"] = [doc.doc_id for doc in final_docs]
    return final_docs, audit, semantic_scores
