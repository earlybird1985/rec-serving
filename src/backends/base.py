"""Backend protocol. Each method takes a batch of request dicts."""
from __future__ import annotations

from typing import Protocol


class Backend(Protocol):
    name: str
    n_items: int
    device: str

    def retrieve_many(self, requests: list[dict]) -> list[dict]:
        ...

    def rank_many(self, requests: list[dict]) -> list[dict]:
        ...

    def generate_many(self, requests: list[dict]) -> list[dict]:
        ...


def ranked_payload(items: list[int], scores: list[float], extra: dict | None = None) -> dict:
    out = {"items": [int(x) for x in items], "scores": [float(s) for s in scores]}
    if extra:
        out.update(extra)
    return out
