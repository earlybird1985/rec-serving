"""Load an A6 HSTU checkpoint. Code comes from a sibling repo or HSTU_SRC."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from backends.base import ranked_payload


def hstu_src_paths() -> list[Path]:
    env = os.environ.get("HSTU_SRC")
    here = Path(__file__).resolve()
    return [
        Path(env) if env else None,
        here.parents[3] / "hstu-recs" / "src",
        here.parents[2] / "vendor" / "hstu",
    ]


def import_hstu():
    for path in hstu_src_paths():
        if path is None or not (path / "hstu.py").exists():
            continue
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        import data as hstu_data
        import hstu as hstu_mod
        import sequences as hstu_seq

        return hstu_mod, hstu_data, hstu_seq
    raise FileNotFoundError(
        "HSTU sources not found. Clone hstu-recs next to this repo or set HSTU_SRC."
    )


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HSTUBackend:
    name = "hstu"

    def __init__(self, checkpoint: str, catalog_chunk: int = 4096):
        hstu_mod, hstu_data, hstu_seq = import_hstu()
        self._mod = hstu_mod
        self._data = hstu_data
        self._seq = hstu_seq
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.n_items = int(ckpt["n_items"])
        self.n_actions = int(ckpt["n_actions"])
        self.device = pick_device()
        self.catalog_chunk = int(catalog_chunk)
        model = hstu_mod.model_from_config(ckpt["cfg"], self.n_items, self.n_actions)
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        model.eval()
        self.model = model

    def _example(self, history: list[int], actions: list[int] | None, times: list[int] | None) -> dict:
        if not history:
            raise ValueError("history_items must be non-empty")
        if actions is None or len(actions) != len(history):
            actions = [0] * len(history)
        if times is None or len(times) != len(history):
            times = list(range(len(history)))
        packed = self._seq.interleave(history, actions, times)
        packed["history_items"] = history
        return packed

    def _encode_last(self, examples: list[dict]) -> torch.Tensor:
        batch = self._data.collate_examples(examples)
        batch = {key: value.to(self.device) for key, value in batch.items()}
        hidden = self.model.encode(batch)
        return self._mod.last_hidden(hidden, batch["mask"])

    @torch.inference_mode()
    def retrieve_many(self, requests: list[dict]) -> list[dict]:
        examples = []
        for req in requests:
            history = [int(x) for x in req["history_items"]]
            examples.append(
                self._example(history, req.get("action_ids"), req.get("timestamps"))
            )
        hidden = self._encode_last(examples)
        weight = self.model.item_emb.weight
        pieces = []
        for start in range(0, weight.size(0), self.catalog_chunk):
            pieces.append(torch.nn.functional.linear(hidden, weight[start : start + self.catalog_chunk]))
        scores = torch.cat(pieces, dim=-1)
        out = []
        for row, req in enumerate(requests):
            history = [int(x) for x in req["history_items"]]
            k = int(req.get("k") or 10)
            row_scores = scores[row].clone()
            for item in history:
                if 0 <= item < self.n_items:
                    row_scores[item] = float("-inf")
            top = torch.topk(row_scores, k=min(k, self.n_items))
            # topk is not stable for ties; sort the kept ids as a second key.
            pairs = sorted(
                [(float(score), int(item)) for score, item in zip(top.values, top.indices)],
                key=lambda p: (-p[0], p[1]),
            )
            out.append(ranked_payload([i for _s, i in pairs], [s for s, _i in pairs]))
        return out

    @torch.inference_mode()
    def rank_many(self, requests: list[dict]) -> list[dict]:
        rows = []
        spans = []
        for req in requests:
            history = [int(x) for x in req["history_items"]]
            cands = [int(x) for x in req.get("candidate_ids") or []]
            base = self._example(history, req.get("action_ids"), req.get("timestamps"))
            start = len(rows)
            for item in cands:
                rows.append(self._data.append_candidate(base, item))
            spans.append((start, len(rows), cands))
        if not rows:
            return [ranked_payload([], []) for _ in requests]
        hidden = self._encode_last(rows)
        high = self.model.action_head(hidden)[:, -1]
        out = []
        for start, end, cands in spans:
            scored = [(float(high[i]), int(cands[i - start])) for i in range(start, end)]
            scored.sort(key=lambda p: (-p[0], p[1]))
            out.append(ranked_payload([i for _s, i in scored], [s for s, _i in scored]))
        return out

    def generate_many(self, requests: list[dict]) -> list[dict]:
        raise RuntimeError("HSTU backend has no /generate. Use retrieve or the beam backend.")
