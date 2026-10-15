# A7 Implementation Plan: Serving the Generative Recommender

> Companion doc to `llm_career_pivot_project_roadmap.md`. This expands A7 into a step-by-step build plan: a batched inference server for the Track A models that are actually in this workspace, plus a latency / quality table (p50 / p99, QPS, Recall@K) under the knobs that change cost.
>
> **A3 and A6 are not Hugging Face causal LMs.** A3 is a from-scratch encoder-decoder over semantic-ID tuples. A6 is a from-scratch HSTU over item ids and rating buckets. [vLLM](https://github.com/vllm-project/vllm) serves HF-compatible generators (continuous batching, PagedAttention). It does not load those two checkpoints. The server here is a small FastAPI process with a micro-batching queue. vLLM is the right tool for A4/A5's Qwen + LoRA; that path is named in §7 and left optional. Do not add `vllm` to `requirements.txt` so a Mac smoke test still installs.
>
> **The roadmap's "beam width" curve is A3's knob.** A6 retrieval is one encoder pass plus a chunked matmul with the item table. A6 ranking is one extra item token per candidate. A4/A5 pointwise scoring is `O(n_cand)` forwards. The bench writes one row per (backend, knob) and keeps a protocol column, the same honesty A6 required for NDCG.

## 0. Scope and success criteria

**What "done" looks like:**

- A process that answers `POST /retrieve`, `POST /rank`, and `POST /generate` and `GET /health`. Request bodies are JSON. Responses include the ranked item ids and a server-side `elapsed_ms`.
- Micro-batching: requests that arrive inside `batch_window_ms` (default 8 ms) share one encoder / beam step, up to `max_batch`. A unit test shows that two retrieve calls that land in the same flush produce the same ranking as two sequential calls.
- Three backends, selected by config:
  - `hstu` — A6 checkpoint (`model.pt` + `n_items` / `n_actions` / `cfg`). Loads code from a sibling `hstu-recs/src` or `HSTU_SRC`. Retrieve = last hidden state dotted with the item table. Rank = A6 `score_candidates` (append one item token, read the action head).
  - `beam` — a tiny fixed-length semantic-ID generator shipped in this repo, plus an optional A3 checkpoint if `a3_checkpoint` is set. Unconstrained and trie-constrained beam search. This is the beam-width axis.
  - `toy` — deterministic vectors, no file. Tests and `serve_small` use this so CI does not need Amazon, Qwen, or an A6 run.
- `src/bench.py` sweeps the cost knobs, writes `checkpoints/<output>/latency.json`, and prints p50 / p95 / p99 ms, QPS, and Recall@K when a `target` is in the request. `RESULTS.md` has the empty table; fill it from that JSON. Do not invent numbers.
- A short write-up that says why a 1.5B Qwen pointwise scorer and a 64-d HSTU are different latency stories, and why vLLM would not have made A6 faster.

**What "done" does NOT require:** sub-100 ms on a laptop (the roadmap's production number is the *target*, not a gate); vLLM, TensorRT, or FlashAttention; serving Qwen in this repo's default path; matching A6's published Recall (the bench may use a toy catalog); rewriting A3/A6 training.

---

## 1. Repo layout

New repo, `rec-serving/` — sits next to `hstu-recs` (A6) and `generative-retrieval` (A3):

```
rec-serving/
├── README.md
├── RESULTS.md
├── requirements.txt
├── configs/
│   ├── serve.yaml                      # hstu backend if a checkpoint exists
│   └── serve_small.yaml                # toy backend, short benches
├── src/
│   ├── utils.py
│   ├── metrics.py                      # percentile, Recall@K, NDCG@K
│   ├── batching.py                     # windowed micro-batch
│   ├── backends/
│   │   ├── base.py                     # retrieve / rank / generate Protocol
│   │   ├── toy.py
│   │   ├── hstu.py                     # load A6; sibling import
│   │   └── beam.py                     # tiny SID generator + A3 hook
│   ├── server.py                       # FastAPI
│   ├── client.py                       # one-request helper
│   └── bench.py                        # sweeps → latency.json
├── notebooks/
│   └── serving.ipynb                   # batching + a fake latency table
├── tests/
│   └── test_serving.py
└── checkpoints/
```

**Papers and code to read (not copy)** before writing the queue:

| Work | What to take | What not to cargo-cult |
|---|---|---|
| [vLLM](https://arxiv.org/abs/2309.06180) / PagedAttention | Continuous batching: a request joins an in-flight batch; KV pages are the memory unit | Installing vLLM for HSTU. There is no HF `generate` here |
| HSTU §3.4 M-FALCON | Score `b_m` candidates in one forward by sharing the history attention | Implementing their fused kernel. We batch `(history, candidate)` rows the obvious way and say so |
| A3 §6 | Beam width vs Recall; unconstrained vs trie; invalid-ID rate | Importing A3's `src/` (it also has `datasets.py` / `utils.py`). Load a checkpoint through `a3_src` if set |
| A4 §11 / A5 §11 | Pointwise is `O(n_cand)` forwards; listwise is one generate. Log wall-clock next to quality | Standing up Qwen behind this FastAPI as the default |

Write `batching.py`, `toy.py`, and `metrics.py` first and unit-test them with no checkpoint.

---

## 2. Why this is not "pip install vLLM"

vLLM's scheduler assumes a causal LM that emits tokens into a paged KV cache. A6's retrieve path does **not** emit tokens. It encodes a right-padded interleaved sequence once, then does `hidden @ E_item.T` in chunks of `catalog_chunk` (default 4096). The quadratic term is `O(L^2 d)` for history length `L` (a few tens after k-core), not `O(new_tokens × N)`. PagedAttention would sit idle.

A3's generate path **does** emit a fixed 5-step tuple. That is closer to vLLM, but the model is not `AutoModelForCausalLM`. The beam lives in `generative-retrieval/src/generate.py`. A7 reimplements the tiny beam and, if `a3_checkpoint` is set, imports that module by file path the same way A4 loaded A1 (`spec_from_file_location`), so A3's `src/` does not shadow this repo.

A4/A5 **are** HF causal LMs. Serving them with vLLM (base + LoRA, prefix cache on the shared history) is the production move. This repo documents the request shape and the `O(n_cand)` cost. It does not download Qwen in the smoke test.

The interview sentence is: the architecture decides the server, not the other way around.

---

## 3. Request / response

All POST bodies are JSON. Histories are item ids, oldest first, already mapped through A6 `item2idx` (or the toy catalog). The server does not tokenize titles.

### `POST /retrieve`

```json
{"history_items": [3, 8, 1], "action_ids": [2, 2, 1], "timestamps": [1, 2, 3], "k": 10, "target": 4}
```

`action_ids` / `timestamps` default to zeros / `0,1,2,...` if omitted. `target` is optional and only used by the bench for Recall.

Response:

```json
{"items": [4, 11, 2], "scores": [1.02, 0.41, 0.33], "elapsed_ms": 3.1, "backend": "hstu"}
```

### `POST /rank`

```json
{"history_items": [3, 8, 1], "candidate_ids": [4, 11, 7, 2], "target": 4}
```

Same response shape. Order is score descending. This is A6's target-aware action head, **not** the reported A6 Recall table (that table dots the full catalog). `RESULTS.md` says so.

### `POST /generate`

```json
{"history_items": [3, 8, 1], "beam_width": 4, "constrained": true, "k": 10, "target": 4}
```

Used by the `beam` backend. `constrained: true` walks the semantic-ID trie. The response `items` are catalog ids; unconstrained beams that miss the trie are dropped and counted in `invalid_rate`.

### `GET /health`

```json
{"ok": true, "backend": "toy", "n_items": 32, "device": "cpu"}
```

---

## 4. Micro-batching (`src/batching.py`)

A retrieve of batch size 1 wastes the GEMM. The server holds requests for up to `batch_window_ms`, then runs `backend.retrieve_many(histories, k)`.

```
enqueue(req) → Future
loop:
    wait until queue non-empty
    wait up to batch_window_ms or until len(queue) == max_batch
    flush: one backend call, set each Future
```

Flush is also triggered when the queue hits `max_batch` before the window ends. Rank and generate have their own queues so a 50-candidate rank does not stall a retrieve.

The unit test does not sleep on a clock. It calls `Batcher.flush()` with two hand-built requests and checks both results.

Do not hide padding bugs behind the batcher. Histories of different lengths are right-padded in the backend, the same as A6 §3.5.

---

## 5. Backends

### 5.1 `toy`

Each item id `i` has a fixed vector `e_i = onehot-ish hash in R^{d}`. A history is the mean of its item vectors. Retrieve ranks by cosine. Rank uses the same score on the candidate subset. Generate walks a pre-built trie of 3-token ids (`n_items = 32`, codebook 4) with a score that prefers the true next item when `target` is provided, otherwise the lowest id. This backend exists so tests and `serve_small` do not need a checkpoint.

### 5.2 `hstu`

```
HSTU_SRC or ../hstu-recs/src  →  model_from_config, collate_examples, interleave,
                                  last_hidden, item_logits, append_candidate
checkpoint                    →  state_dict, cfg, n_items, n_actions
```

`retrieve_many`:

1. Interleave each history (A6 Table 11 layout, no labels required at serve time).
2. Collate, `model.encode`, `last_hidden`.
3. `item_logits` in chunks of `catalog_chunk`.
4. Set history ids to `−∞`. Argsort, take `k`. Ties break toward the smaller item id.

`rank_many`: for each request, `append_candidate` every id, one batched encode, action-head logit of bucket `n_actions - 1` (the "high" bucket). That is A6's `candidate_action_scores`. It is **not** M-FALCON: the history is re-encoded per candidate row. Log both `n_cand` and `elapsed_ms` so the quadratic waste is visible.

If the checkpoint is missing, the process exits with a path in the error. `serve.yaml` may point at `../hstu-recs/checkpoints/hstu/model.pt`; `serve_small.yaml` uses `toy`.

### 5.3 `beam`

Shipped `TinySID`: embedding of the last history item, then `P` linear heads (default `P=3`, `K=4`). Beam search copies A3's loop: at each level expand `beam_width`, keep the top `beam_width` prefixes. Constrained mode uses `SemanticIDTrie` over a `semid2item` dict.

Optional A3: if `a3_checkpoint` and `a3_src` are set, load `GenerativeRetriever` + `SemanticIDVocab` via `importlib.util.spec_from_file_location` (same reason as A4's `load_two_tower`: A3's `src/` also has `utils.py`). Then `/generate` is real A3. `/retrieve` on this backend is an error unless you define it as generate-with-default-beam; do not silently call HSTU.

---

## 6. Bench (`src/bench.py`)

The bench is a client. It does not import the model into the same process as a second copy if `--base-url` is set; it HTTP-posts. `--in-process` uses the backend directly so `tests/` and a laptop without two terminals still run.

Sweeps, from the config:

| backend | knob | default values |
|---|---|---|
| hstu / toy retrieve | `k` | 10, 50, 100 |
| hstu / toy retrieve | `batch_window_ms` | 0, 8, 16 |
| hstu / toy retrieve | `catalog_chunk` | 1024, 4096 (hstu only) |
| hstu / toy rank | `n_cand` | 10, 50, 100 |
| beam generate | `beam_width` | 1, 2, 4, 8 |
| beam generate | `constrained` | true, false |

For each cell: `warmup` requests, then `n_requests` timed requests. Report:

- `p50_ms`, `p95_ms`, `p99_ms` of end-to-end latency (client clock if HTTP, `perf_counter` if in-process)
- `qps = n_requests / wall_s`
- `recall@k` / `ndcg@k` when every request carried a `target`
- `invalid_rate` on unconstrained beam
- `backend`, `device`, `n_items`, `max_batch`, `batch_window_ms`

`batch_window_ms: 0` means flush every request (batch size 1). That row is the "I forgot to batch" baseline.

Payloads: if `user_sequences.parquet` from A6 exists, take the test prefix (`items[:-1]`) and `target = items[-1]`, same leave-last-out as A6. Else synthesize `n_users` histories from the toy catalog.

Write `checkpoints/<output>/latency.json`. `RESULTS.md` copies from there. Small-config numbers stay in a separate section and are not compared to a GPU A6 run.

---

## 7. Optional: A4/A5 and vLLM

Not built in the default server.

If you later serve Qwen:

- vLLM (or TGI) loads `Qwen/Qwen2.5-1.5B-Instruct` plus the A4/A5 adapter.
- The shared history is a prefix; pointwise scoring still needs `n_cand` forwards unless you pack candidates into one batched prompt.
- Listwise `generate` is one call but a long decode; p99 will track `max_new_tokens`, not `n_cand`.
- 4-bit is CUDA-only, same Darwin rule as A4/A5.

Put those rows in `RESULTS.md` only after a real vLLM run. This repo's table is HSTU / toy / tiny-beam.

---

## 8. Milestone checklist

- [ ] `tests/test_serving.py` passes with no checkpoint: toy retrieve/rank/generate, batcher flush, unconstrained invalid rate > 0, constrained invalid rate = 0, percentile helper.
- [ ] `python src/server.py --config configs/serve_small.yaml` then `GET /health` returns `backend: toy`.
- [ ] `python src/bench.py --config configs/serve_small.yaml --in-process` writes `latency.json` with p99 and a beam-width sweep.
- [ ] With an A6 `model.pt`, `serve.yaml` + `/retrieve` returns `k` items and excludes history ids.
- [ ] `RESULTS.md` filled from that JSON, including the sentence that rank-endpoint NDCG is not A6's full-catalog number, and that vLLM was not used.

---

## 9. Common failure modes

| Symptom | Likely cause | What to do |
|---|---|---|
| p99 ≫ p50 on batch size 1 | First-request CUDA / MPS warmup counted | `warmup >= 5`; report only the timed window |
| Batched retrieve ≠ sequential | Left-pad, or history exclusion applied to the wrong row | Right-pad; mask per row; unit test in §4 |
| `/retrieve` includes a history item | Forgot `−∞` on history ids | Same as A6 eval |
| HSTU import fails | Ran from a clone without `hstu-recs` next door | Set `HSTU_SRC`; or use `toy` |
| Constrained invalid_rate ≠ 0 | Trie built from a different `semid2item` than the decoder vocab | Rebuild the trie from the same dict the backend uses |
| Unconstrained invalid_rate = 0 on the toy | Trie leaked into the unconstrained path | The test expects a nonzero rate on a sparse codebook |
| "vLLM is slow on HSTU" | vLLM is not in this stack | Do not put that sentence in `RESULTS.md` |
| Rank p99 linear in `n_cand` | Expected: we re-encode history per candidate | Note it; M-FALCON is the paper's fix, not this repo's |

---

## 10. Suggested order of work

1. **Day 1:** `metrics.py`, `batching.py`, `backends/toy.py`, `tests/test_serving.py`. No server.
2. **Day 2:** FastAPI routes + `client.py`. Health + one retrieve through HTTP.
3. **Day 3:** `bench.py --in-process` on `serve_small`. Beam-width and `n_cand` sweeps. Read the JSON by hand.
4. **Day 4:** `backends/hstu.py` against a sibling A6 checkpoint if it exists. Confirm history ids are dropped.
5. **Day 5:** Optional A3 hook. If no A3 checkpoint, leave `a3_checkpoint: null` and keep the tiny SID.
6. **Day 6:** `RESULTS.md`. Latency vs Recall for toy/HSTU. One paragraph on why the Qwen reranker is a different server.

A couple of evenings if A6 already has a `model.pt`. The smoke path does not wait on that.

---

## 11. What this closes in Track A

- A1–A3 retrieve from a catalog. A7 times that retrieve when it is a model forward plus a GEMM or a 5-step beam, not a FAISS script.
- A4–A5 rerank `n_cand` titles. A7's rank sweep is the same cost shape on A6's action head, which is enough to talk about `O(n_cand)` vs a single generate.
- A6 handed off a `state_dict`. This repo is the process that loads it.
- Sub-100 ms p99 on a 1.5B listwise decoder is still the reason generative recs stay offline. Write the number you measured, even if it is 800 ms on a laptop.

---

## 12. References

1. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. https://arxiv.org/abs/2309.06180 — vLLM, continuous batching.
2. Zhai et al. *Actions Speak Louder than Words.* 2024. https://arxiv.org/abs/2402.17152 — §3.4 M-FALCON, candidate microbatches.
3. Roadmap A3 §6 — beam width, trie, invalid-ID rate.
4. Roadmap A4 §11 / A5 §11 — pointwise `O(n_cand)` vs listwise generate; log wall-clock.
5. Roadmap A6 §11 — HSTU checkpoint is a `state_dict`; do not load an A5 adapter as this model.
6. Roadmap A7 — latency / quality curve; why generative recs are still mostly not in production.
