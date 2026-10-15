"""Tiny semantic-ID generator, plus an optional A3 checkpoint hook."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn

from backends.base import ranked_payload
from backends.trie import SemanticIDTrie, beam_search
from backends.toy import item_semid


class TinySID(nn.Module):
    """Last-item embedding plus one linear head per SID position."""

    def __init__(self, n_items: int, n_levels: int = 3, codebook: int = 4, d_model: int = 16):
        super().__init__()
        self.n_items = n_items
        self.n_levels = n_levels
        self.codebook = codebook
        self.embed = nn.Embedding(n_items, d_model)
        self.heads = nn.ModuleList([nn.Linear(d_model, codebook) for _ in range(n_levels)])

    def prefix_logits(self, last_item: int, prefix: list[int]) -> list[float]:
        hidden = self.embed(torch.tensor([int(last_item) % self.n_items]))
        level = min(len(prefix), self.n_levels - 1)
        return self.heads[level](hidden)[0].tolist()


class BeamBackend:
    name = "beam"

    def __init__(
        self,
        n_items: int = 32,
        n_levels: int = 3,
        codebook: int = 4,
        a3_checkpoint: str | None = None,
        a3_src: str | None = None,
    ):
        self.n_items = n_items
        self.n_levels = n_levels
        self.codebook = codebook
        self.device = "cpu"
        self.semid2item = {item_semid(i, n_levels, codebook): i for i in range(n_items)}
        self.trie = SemanticIDTrie(self.semid2item)
        self.tiny = TinySID(n_items, n_levels, codebook)
        self.tiny.eval()
        self._a3 = None
        if a3_checkpoint:
            self._a3 = load_a3(a3_checkpoint, a3_src)

    def retrieve_many(self, requests: list[dict]) -> list[dict]:
        for req in requests:
            req = dict(req)
            req.setdefault("beam_width", req.get("k") or 10)
            req.setdefault("constrained", True)
        return self.generate_many(requests)

    def rank_many(self, requests: list[dict]) -> list[dict]:
        raise RuntimeError("beam backend has no /rank. Use /generate or the hstu backend.")

    def generate_many(self, requests: list[dict]) -> list[dict]:
        if self._a3 is not None:
            return self._a3.generate_many(requests)
        out = []
        for req in requests:
            history = [int(x) for x in req.get("history_items") or [0]]
            last = history[-1] if history else 0
            width = int(req.get("beam_width") or 4)
            constrained = bool(req.get("constrained", True))
            k = int(req.get("k") or width)

            def logits(prefix: list[int]) -> list[float]:
                return self.tiny.prefix_logits(last, prefix)

            beams, invalid = beam_search(
                logits,
                self.n_levels,
                self.codebook,
                width,
                self.trie if constrained else None,
                self.semid2item,
            )
            items = [item for item, _s in beams[:k] if item >= 0]
            scores = [s for item, s in beams[:k] if item >= 0]
            out.append(ranked_payload(items, scores, {"invalid_rate": invalid, "n_beams": len(beams)}))
        return out


def load_a3(checkpoint: str, a3_src: str | None):
    """Import A3 by file path so its src/utils.py does not shadow this repo."""
    src = Path(a3_src or Path(__file__).resolve().parents[3] / "generative-retrieval" / "src")
    if not (src / "generate.py").exists():
        raise FileNotFoundError(f"A3 sources not found at {src}")

    def load(name: str):
        spec = importlib.util.spec_from_file_location(f"a3_{name}", src / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    generate = load("generate")
    model_mod = load("model")
    vocab_mod = load("vocab")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # A3 checkpoint layout is whatever that repo saved; keep this hook explicit.
    raise RuntimeError(
        f"A3 checkpoint hook is present ({checkpoint}, {model_mod.__name__}, "
        f"{vocab_mod.__name__}, {generate.__name__}) but the saved key layout "
        "is not wired. Point serve.yaml at toy/hstu until an A3 run is exported."
    )
