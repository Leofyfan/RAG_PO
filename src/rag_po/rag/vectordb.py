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
