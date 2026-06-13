from __future__ import annotations

from rag_po.defense.social_rerank import social_credibility_score
from rag_po.math_utils import cosine_similarity
from rag_po.models import TweetDocument


def trust_aware_dedup(
    docs: list[TweetDocument],
    threshold: float = 0.90,
) -> tuple[list[TweetDocument], dict[str, object]]:
    """Cluster near-duplicate retrieved docs and keep the most trusted representative."""
    clusters: list[list[TweetDocument]] = []
    for doc in docs:
        placed = False
        for cluster in clusters:
            if any(cosine_similarity(doc.embedding, other.embedding) >= threshold for other in cluster):
                cluster.append(doc)
                placed = True
                break
        if not placed:
            clusters.append([doc])

    kept: list[TweetDocument] = []
    audit_clusters: list[dict[str, object]] = []
    for cluster in clusters:
        ranked = sorted(cluster, key=social_credibility_score, reverse=True)
        representative = ranked[0]
        kept.append(representative)
        audit_clusters.append(
            {
                "representative_id": representative.doc_id,
                "member_ids": [doc.doc_id for doc in cluster],
                "size": len(cluster),
                "coordinated_injection": len(cluster) >= 3,
                "trust_scores": {doc.doc_id: social_credibility_score(doc) for doc in cluster},
            }
        )
    return kept, {"clusters": audit_clusters, "removed_count": len(docs) - len(kept)}
