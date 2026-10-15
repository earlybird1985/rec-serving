# Serving the Generative Recommender

A batched inference server for Track A models, plus a latency / quality bench.

This is **Project A7** of a larger LLM-for-recommendation learning roadmap —
see [`A7_serving_implementation.md`](A7_serving_implementation.md) for why
vLLM does not load A3 or A6, how micro-batching works, and which knob
(beam width vs `n_cand` vs `k`) belongs on which backend.

Default smoke backend is `toy` (no checkpoint). Point `serve.yaml` at an
A6 `model.pt` and set `HSTU_SRC` (or keep `hstu-recs` next to this repo)
to serve a real HSTU retrieve.

## Status

- [x] Toy retrieve / rank / generate and a windowed batcher
- [x] FastAPI routes and an in-process bench
- [x] HSTU backend that imports A6 sources from a sibling path
- [x] Tiny semantic-ID beam (constrained vs unconstrained)
- [x] Unit tests with no download
- [ ] HTTP bench against a running server
- [ ] `RESULTS.md` filled from a real A6 checkpoint

## Repo layout

```
├── configs/serve.yaml           # hstu if the A6 checkpoint exists
├── configs/serve_small.yaml     # toy, short sweeps
├── src/server.py
├── src/bench.py
├── src/batching.py
├── src/backends/
└── tests/test_serving.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

From the repo root:

```bash
python tests/test_serving.py

python src/bench.py --config configs/serve_small.yaml --in-process

python src/server.py --config configs/serve_small.yaml
# in another shell
python -c "from src.client import get; print(get('http://127.0.0.1:8000', '/health'))"
```

HSTU retrieve, after A6 has written a checkpoint:

```bash
export HSTU_SRC=../hstu-recs/src
python src/server.py --config configs/serve.yaml
python src/bench.py --config configs/serve.yaml --in-process
```

## Design notes

- **The architecture picks the server.** A6 retrieve is one encode plus a
  catalog GEMM. A3 is a 5-step beam. A4/A5 are HF causal LMs (vLLM's job,
  not this process).
- **Right pad, drop history ids**, same as A6 eval.
- **Rank re-encodes each candidate.** That is the simple batch, not M-FALCON.
- **Do not invent RESULTS numbers.** Copy `latency.json`.
