"""Latency / quality sweeps. Writes checkpoints/<output>/latency.json.

    python src/bench.py --config configs/serve_small.yaml --in-process
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends.factory import build_backend
from batching import timed_ms
from metrics import ndcg_at_k, percentile, recall_at_k
from utils import load_config, save_json


def load_payloads(cfg: dict, backend) -> list[dict]:
    parquet = cfg.get("user_sequences")
    n_users = int(cfg.get("n_bench_users", 32))
    if parquet and Path(parquet).exists():
        import pandas as pd

        rows = []
        frame = pd.read_parquet(parquet)
        for rec in frame.itertuples(index=False):
            items = [int(x) for x in rec.item_idxs]
            if len(items) < 3:
                continue
            actions = [int(x) for x in rec.action_ids] if hasattr(rec, "action_ids") else [0] * len(items)
            times = [int(x) for x in rec.timestamps] if hasattr(rec, "timestamps") else list(range(len(items)))
            rows.append(
                {
                    "history_items": items[:-1],
                    "action_ids": actions[:-1],
                    "timestamps": times[:-1],
                    "target": items[-1],
                    "candidate_ids": [items[-1]] + [x for x in items[:-1][:4]],
                }
            )
            if len(rows) >= n_users:
                break
        if rows:
            return rows
    payloads = []
    n_items = getattr(backend, "n_items", 32)
    for i in range(n_users):
        history = [((i * 3 + j) % n_items) for j in range(3)]
        target = (history[-1] + 1) % n_items
        payloads.append(
            {
                "history_items": history,
                "target": target,
                "candidate_ids": [target, (target + 2) % n_items, (target + 5) % n_items],
            }
        )
    return payloads


def summarize(latencies: list[float], recalls: list[float], ndcgs: list[float], invalids: list[float], wall_s: float, extra: dict) -> dict:
    n = len(latencies)
    return {
        **extra,
        "n": n,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "qps": (n / wall_s) if wall_s > 0 else 0.0,
        "recall": sum(recalls) / len(recalls) if recalls else None,
        "ndcg": sum(ndcgs) / len(ndcgs) if ndcgs else None,
        "invalid_rate": sum(invalids) / len(invalids) if invalids else None,
    }


def run_cell(call, payloads: list[dict], warmup: int, k: int) -> dict:
    for req in payloads[:warmup]:
        call(req)
    latencies = []
    recalls = []
    ndcgs = []
    invalids = []
    start = time.perf_counter()
    for req in payloads:
        result, elapsed = timed_ms(lambda r=req: call(r))
        latencies.append(elapsed)
        target = req.get("target")
        items = result.get("items") or []
        rec = recall_at_k(items, target, k)
        ndcg = ndcg_at_k(items, target, k)
        if rec is not None:
            recalls.append(rec)
        if ndcg is not None:
            ndcgs.append(ndcg)
        if "invalid_rate" in result:
            invalids.append(float(result["invalid_rate"]))
    wall = time.perf_counter() - start
    return summarize(latencies, recalls, ndcgs, invalids, wall, {})


def in_process_caller(backend, kind: str):
    def call(req: dict) -> dict:
        if kind == "retrieve":
            return backend.retrieve_many([req])[0]
        if kind == "rank":
            return backend.rank_many([req])[0]
        if kind == "generate":
            return backend.generate_many([req])[0]
        raise ValueError(kind)

    return call


def http_caller(base_url: str, kind: str):
    from client import post

    path = {"retrieve": "/retrieve", "rank": "/rank", "generate": "/generate"}[kind]

    def call(req: dict) -> dict:
        return post(base_url, path, req)

    return call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    cfg = load_config(args.config)
    backend = build_backend(cfg)
    payloads = load_payloads(cfg, backend)
    warmup = int(cfg.get("warmup", 2))
    rows = []

    def add(kind: str, knob: dict, call):
        k = int(knob.get("k") or cfg.get("k", 10))
        stats = run_cell(call, payloads, warmup, k)
        stats.update({"kind": kind, "backend": backend.name, "device": backend.device, **knob})
        rows.append(stats)
        print(stats)

    caller = in_process_caller(backend, "retrieve") if args.in_process else http_caller(args.base_url, "retrieve")
    for k in cfg.get("k_sweep", [10]):
        add("retrieve", {"k": int(k)}, caller)

    if backend.name in {"toy", "hstu"}:
        rank_call = in_process_caller(backend, "rank") if args.in_process else http_caller(args.base_url, "rank")
        for n_cand in cfg.get("n_cand_sweep", [10]):
            def rank_with_n(req, n=int(n_cand)):
                body = dict(req)
                cands = list(body.get("candidate_ids") or [])
                while len(cands) < n:
                    cands.append((cands[-1] + 1) % backend.n_items if cands else 0)
                body["candidate_ids"] = cands[:n]
                return rank_call(body)

            add("rank", {"n_cand": int(n_cand), "k": int(n_cand)}, rank_with_n)

    if backend.name in {"toy", "beam"}:
        gen_call = in_process_caller(backend, "generate") if args.in_process else http_caller(args.base_url, "generate")
        for width in cfg.get("beam_width_sweep", [1, 4]):
            for constrained in cfg.get("constrained_sweep", [True, False]):
                def gen(req, w=int(width), c=bool(constrained)):
                    body = dict(req)
                    body["beam_width"] = w
                    body["constrained"] = c
                    body["k"] = w
                    return gen_call(body)

                add("generate", {"beam_width": int(width), "constrained": bool(constrained), "k": int(width)}, gen)

    output_dir = Path(cfg.get("output_dir") or "checkpoints/serve")
    path = output_dir / "latency.json"
    save_json(
        {
            "protocol": "serving latency; Recall is of the served list, not A6 full-catalog unless backend=hstu retrieve",
            "backend": backend.name,
            "n_items": backend.n_items,
            "in_process": bool(args.in_process),
            "rows": rows,
        },
        path,
    )
    print("wrote", path)


if __name__ == "__main__":
    main()
