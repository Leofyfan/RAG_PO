from __future__ import annotations

import math
from statistics import mean, pstdev


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return dot(a, b) / denom


def cosine_distance(a: list[float] | None, b: list[float] | None) -> float:
    return 1.0 - cosine_similarity(a, b)


def z_threshold(values: list[float], z: float) -> float:
    if not values:
        return 0.0
    return mean(values) + z * pstdev(values)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
