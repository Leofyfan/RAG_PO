from __future__ import annotations

import math
import random

from rag_po.models import TweetDocument


def injection_count(clean_size: int, ratio: float, available: int) -> int:
    if ratio <= 0 or clean_size <= 0 or available <= 0:
        return 0
    return min(available, max(1, math.ceil(clean_size * ratio)))


def build_random_poisoned_corpus(
    clean_docs: list[TweetDocument],
    rumour_pool: list[TweetDocument],
    ratio: float,
    seed: int = 13,
) -> list[TweetDocument]:
    n = injection_count(len(clean_docs), ratio, len(rumour_pool))
    rng = random.Random(seed)
    selected = rng.sample(rumour_pool, n) if n else []
    poisoned = [doc.with_attack("random") for doc in selected]
    return list(clean_docs) + poisoned
