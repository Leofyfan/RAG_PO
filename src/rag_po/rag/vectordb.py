from __future__ import annotations

from rag_po.math_utils import cosine_similarity
from rag_po.models import SearchResult, TweetDocument


class InMemoryVectorStore:
    """Small vector store with cosine search; works without external services."""

    def __init__(self, docs: list[TweetDocument]):
        self.docs = list(docs)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        scored = [
            SearchResult(doc=doc, score=cosine_similarity(query_embedding, doc.embedding), rank=0)
            for doc in self.docs
            if doc.embedding is not None
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return [SearchResult(doc=item.doc, score=item.score, rank=i + 1) for i, item in enumerate(scored[:top_k])]

    def search_mmr(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        lambda_mult: float = 0.7,
        fetch_k: int | None = None,
    ) -> list[SearchResult]:
        candidates = self.search(query_embedding, top_k=fetch_k or max(top_k * 4, top_k))
        selected: list[SearchResult] = []
        remaining = list(candidates)
        while remaining and len(selected) < top_k:
            if not selected:
                selected.append(remaining.pop(0))
                continue
            best_index = 0
            best_score = float("-inf")
            best_penalty = float("inf")
            for idx, candidate in enumerate(remaining):
                diversity_penalty = max(
                    cosine_similarity(candidate.doc.embedding, item.doc.embedding) for item in selected
                )
                mmr_score = lambda_mult * candidate.score - (1.0 - lambda_mult) * diversity_penalty
                if mmr_score > best_score + 1e-12 or (
                    abs(mmr_score - best_score) <= 1e-12 and diversity_penalty < best_penalty
                ):
                    best_score = mmr_score
                    best_penalty = diversity_penalty
                    best_index = idx
            selected.append(remaining.pop(best_index))
        return [SearchResult(doc=item.doc, score=item.score, rank=i + 1) for i, item in enumerate(selected)]
