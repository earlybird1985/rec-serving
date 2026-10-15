"""Deterministic catalog. No checkpoint. Tests and serve_small use this."""
from __future__ import annotations

import math

from backends.base import ranked_payload
from backends.trie import SemanticIDTrie, beam_search


def item_semid(item: int, n_levels: int = 3, codebook: int = 4) -> tuple[int, ...]:
    codes = []
    value = int(item)
    for _ in range(n_levels):
        codes.append(value % codebook)
        value //= codebook
    return tuple(codes)


def history_vector(history: list[int], dim: int, n_items: int) -> list[float]:
    vec = [0.0] * dim
    if not history:
        return vec
    for item in history:
        idx = int(item) % n_items
        vec[idx % dim] += 1.0
        vec[(idx * 3) % dim] += 0.5
    scale = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / scale for x in vec]


def item_vector(item: int, dim: int, n_items: int) -> list[float]:
    return history_vector([int(item)], dim, n_items)


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class ToyBackend:
    name = "toy"

    def __init__(self, n_items: int = 32, dim: int = 16, n_levels: int = 3, codebook: int = 4):
        self.n_items = n_items
        self.dim = dim
        self.n_levels = n_levels
        self.codebook = codebook
        self.device = "cpu"
        self.semid2item = {item_semid(i, n_levels, codebook): i for i in range(n_items)}
        self.trie = SemanticIDTrie(self.semid2item)

    def _score_catalog(self, history: list[int], banned: set[int]) -> list[tuple[float, int]]:
        query = history_vector(history, self.dim, self.n_items)
        scored = []
        for item in range(self.n_items):
            if item in banned:
                continue
            scored.append((dot(query, item_vector(item, self.dim, self.n_items)), item))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return scored

    def retrieve_many(self, requests: list[dict]) -> list[dict]:
        out = []
        for req in requests:
            history = [int(x) for x in req.get("history_items") or []]
            k = int(req.get("k") or 10)
            banned = set(history)
            ranked = self._score_catalog(history, banned)[:k]
            out.append(ranked_payload([i for _s, i in ranked], [s for s, _i in ranked]))
        return out

    def rank_many(self, requests: list[dict]) -> list[dict]:
        out = []
        for req in requests:
            history = [int(x) for x in req.get("history_items") or []]
            cands = [int(x) for x in req.get("candidate_ids") or []]
            query = history_vector(history, self.dim, self.n_items)
            scored = [(dot(query, item_vector(c, self.dim, self.n_items)), c) for c in cands]
            scored.sort(key=lambda pair: (-pair[0], pair[1]))
            out.append(ranked_payload([i for _s, i in scored], [s for s, _i in scored]))
        return out

    def generate_many(self, requests: list[dict]) -> list[dict]:
        out = []
        for req in requests:
            width = int(req.get("beam_width") or 4)
            constrained = bool(req.get("constrained", True))
            k = int(req.get("k") or width)
            target = req.get("target")
            preferred = item_semid(int(target), self.n_levels, self.codebook) if target is not None else None

            def logits(prefix: list[int]) -> list[float]:
                scores = [0.0] * self.codebook
                if preferred is not None and len(prefix) < self.n_levels:
                    scores[preferred[len(prefix)]] = 2.0
                return scores

            beams, invalid = beam_search(
                logits,
                self.n_levels,
                self.codebook,
                width,
                self.trie if constrained else None,
                self.semid2item,
            )
            items = [item for item, _score in beams[:k] if item >= 0]
            scores = [score for item, score in beams[:k] if item >= 0]
            out.append(
                ranked_payload(items, scores, {"invalid_rate": invalid, "n_beams": len(beams)})
            )
        return out
