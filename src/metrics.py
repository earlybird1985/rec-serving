"""Percentiles and single-relevant-item Recall / NDCG."""
from __future__ import annotations

import math


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. q is in [0, 100]. Empty list is 0."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if q <= 0:
        return ordered[0]
    if q >= 100:
        return ordered[-1]
    index = math.ceil(q / 100.0 * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def recall_at_k(ranked: list[int], target: int | None, k: int) -> float | None:
    if target is None:
        return None
    return 1.0 if int(target) in [int(x) for x in ranked[:k]] else 0.0


def ndcg_at_k(ranked: list[int], target: int | None, k: int) -> float | None:
    if target is None:
        return None
    try:
        rank = [int(x) for x in ranked[:k]].index(int(target))
    except ValueError:
        return 0.0
    return 1.0 / math.log2(rank + 2)
