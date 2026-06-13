from __future__ import annotations

from rag_po.defense.claim_extract import trust_weighted_claim_rank
from rag_po.defense.consistency import consistency_filter
from rag_po.defense.dedup import trust_aware_dedup
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
    actual_retrieve_k = max(retrieve_k, 18) if spec.mmr else retrieve_k
    if spec.mmr:
        initial_results = store.search_mmr(query_embedding, top_k=actual_retrieve_k, fetch_k=max(actual_retrieve_k * 3, 36))
    else:
        initial_results = store.search(query_embedding, top_k=actual_retrieve_k)
    docs = [item.doc for item in initial_results]
    semantic_scores = {item.doc.doc_id: item.score for item in initial_results}
    audit: dict[str, object] = {
        "initial_ids": [doc.doc_id for doc in docs],
        "initial_scores": semantic_scores,
        "defense": defense,
        "retrieval_mode": "mmr" if spec.mmr else "topk",
        "retrieve_k": actual_retrieve_k,
    }

    if spec.outlier:
        docs, outlier_audit = semantic_outlier_filter(docs)
        audit["outlier"] = outlier_audit
    if spec.dedup:
        docs, dedup_audit = trust_aware_dedup(docs)
        audit["dedup"] = dedup_audit
    if spec.consistency:
        docs, consistency_audit = consistency_filter(docs)
        audit["consistency"] = consistency_audit
    if spec.social:
        docs, social_audit = social_rerank_with_audit(docs, semantic_scores)
        audit["social"] = social_audit
    if spec.trust_weighted:
        cluster_sizes: dict[str, int] = {}
        for cluster in (audit.get("dedup") or {}).get("clusters", []):  # type: ignore[union-attr]
            size = int(cluster.get("size", 1))
            for doc_id in cluster.get("member_ids", []):
                cluster_sizes[str(doc_id)] = size
            cluster_sizes[str(cluster.get("representative_id"))] = size
        docs, claim_audit = trust_weighted_claim_rank(docs, semantic_scores, cluster_sizes=cluster_sizes)
        audit["claim_aggregation"] = claim_audit

    final_docs = docs[:final_k]
    audit["final_ids"] = [doc.doc_id for doc in final_docs]
    return final_docs, audit, semantic_scores
