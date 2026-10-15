"""FastAPI process. Micro-batches retrieve / rank / generate separately.

    python src/server.py --config configs/serve_small.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends.factory import build_backend
from batching import Batcher
from utils import load_config


class RetrieveBody(BaseModel):
    history_items: list[int]
    action_ids: list[int] | None = None
    timestamps: list[int] | None = None
    k: int = 10
    target: int | None = None


class RankBody(BaseModel):
    history_items: list[int]
    candidate_ids: list[int]
    action_ids: list[int] | None = None
    timestamps: list[int] | None = None
    target: int | None = None


class GenerateBody(BaseModel):
    history_items: list[int] = Field(default_factory=list)
    beam_width: int = 4
    constrained: bool = True
    k: int = 10
    target: int | None = None


def create_app(cfg: dict) -> FastAPI:
    backend = build_backend(cfg)
    max_batch = int(cfg.get("max_batch", 16))
    window = float(cfg.get("batch_window_ms", 8))
    retrieve_q = Batcher(backend.retrieve_many, max_batch=max_batch, window_ms=window)
    rank_q = Batcher(backend.rank_many, max_batch=max_batch, window_ms=window)
    generate_q = Batcher(backend.generate_many, max_batch=max_batch, window_ms=window)
    app = FastAPI(title="rec-serving")
    app.state.backend = backend

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "backend": backend.name,
            "n_items": backend.n_items,
            "device": backend.device,
        }

    async def _run(batcher: Batcher, payload: dict) -> dict:
        start = time.perf_counter()
        try:
            result = await batcher.submit(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result["elapsed_ms"] = (time.perf_counter() - start) * 1000.0
        result["backend"] = backend.name
        return result

    @app.post("/retrieve")
    async def retrieve(body: RetrieveBody):
        return await _run(retrieve_q, body.model_dump())

    @app.post("/rank")
    async def rank(body: RankBody):
        return await _run(rank_q, body.model_dump())

    @app.post("/generate")
    async def generate(body: GenerateBody):
        return await _run(generate_q, body.model_dump())

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    app = create_app(cfg)
    import uvicorn

    uvicorn.run(app, host=args.host, port=int(cfg.get("port", args.port)))


if __name__ == "__main__":
    main()
