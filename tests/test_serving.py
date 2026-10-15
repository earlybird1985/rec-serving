"""Batching, toy retrieve/rank/generate, percentiles. No checkpoint."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from backends.factory import build_backend
from backends.toy import ToyBackend, item_semid
from backends.trie import SemanticIDTrie, beam_search
from batching import Batcher
from metrics import ndcg_at_k, percentile, recall_at_k


def test_percentile_nearest_rank():
    assert percentile([], 99) == 0.0
    assert percentile([10], 50) == 10
    assert percentile([1, 2, 3, 4], 100) == 4
    assert percentile([1, 2, 3, 4], 50) == 2


def test_single_relevant_metrics():
    assert recall_at_k([4, 1, 2], 4, 10) == 1.0
    assert recall_at_k([1, 2], 4, 10) == 0.0
    assert ndcg_at_k([4], 4, 10) == 1.0
    assert abs(ndcg_at_k([1, 4], 4, 10) - 1.0 / math.log2(3)) < 1e-9
    assert recall_at_k([1], None, 10) is None


def test_batcher_flush_matches_one_by_one():
    backend = ToyBackend(n_items=16)
    reqs = [
        {"history_items": [1, 2], "k": 5},
        {"history_items": [3, 4, 5], "k": 5},
    ]
    batcher = Batcher(backend.retrieve_many, max_batch=8, window_ms=0)
    together = batcher.flush(reqs)
    separate = [backend.retrieve_many([req])[0] for req in reqs]
    assert [row["items"] for row in together] == [row["items"] for row in separate]


def test_retrieve_excludes_history():
    backend = ToyBackend(n_items=16)
    history = [2, 7]
    items = backend.retrieve_many([{"history_items": history, "k": 16}])[0]["items"]
    assert 2 not in items
    assert 7 not in items


def test_rank_orders_the_given_candidates():
    backend = ToyBackend(n_items=16)
    cands = [3, 8, 1]
    items = backend.rank_many([{"history_items": [0, 1], "candidate_ids": cands}])[0]["items"]
    assert set(items) == set(cands)


def test_constrained_beam_never_hallucinates():
    backend = ToyBackend(n_items=32)
    row = backend.generate_many(
        [{"history_items": [1], "beam_width": 8, "constrained": True, "k": 8}]
    )[0]
    assert row["invalid_rate"] == 0.0
    assert all(0 <= item < 32 for item in row["items"])


def test_unconstrained_beam_can_miss_the_trie():
    n_items, n_levels, codebook = 8, 3, 4
    semid2item = {item_semid(i, n_levels, codebook): i for i in range(n_items)}
    trie = SemanticIDTrie(semid2item)

    def uniform(_prefix):
        return [0.0] * codebook

    _constrained, c_rate = beam_search(uniform, n_levels, codebook, 16, trie, semid2item)
    unconstrained, u_rate = beam_search(uniform, n_levels, codebook, 16, None, semid2item)
    assert c_rate == 0.0
    assert u_rate > 0.0


def test_factory_toy():
    backend = build_backend({"backend": "toy", "n_items": 8})
    assert backend.name == "toy"
    assert backend.n_items == 8


if __name__ == "__main__":
    test_percentile_nearest_rank()
    test_single_relevant_metrics()
    test_batcher_flush_matches_one_by_one()
    test_retrieve_excludes_history()
    test_rank_orders_the_given_candidates()
    test_constrained_beam_never_hallucinates()
    test_unconstrained_beam_can_miss_the_trie()
    test_factory_toy()
    print("test_serving: ok")
