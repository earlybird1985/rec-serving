# Results

Not written yet. Fill this in after `src/bench.py` writes `checkpoints/serve/latency.json` (or `serve_small` for a smoke run). Small-config numbers are not comparable to a GPU A6 retrieve.

Protocol: end-to-end latency of one served request. Recall@K / NDCG@K are computed on the **returned list**. For `hstu` `/retrieve` that list is a full-catalog top-K (history ids removed), so it matches A6's metric. For `/rank` it is only the candidate subset. For the toy / tiny-beam backends it is not an A3/A6 number.

vLLM was not used. A3 and A6 are not Hugging Face causal LMs.

## Setup

| | |
|---|---|
| backend | |
| device | |
| n_items | |
| max_batch | |
| batch_window_ms | |
| in-process vs HTTP | |

## Retrieve (HSTU / toy)

| k | p50 ms | p95 ms | p99 ms | QPS | Recall | NDCG |
|---|---|---|---|---|---|---|
| 10 | | | | | | |
| 50 | | | | | | |
| 100 | | | | | | |

## Rank vs n_cand

History is re-encoded per candidate. Linear growth is expected.

| n_cand | p50 ms | p99 ms | QPS |
|---|---|---|---|
| 10 | | | |
| 50 | | | |

## Generate vs beam width (tiny SID / A3)

| beam_width | constrained | p50 ms | p99 ms | Recall | invalid_rate |
|---|---|---|---|---|---|
| 1 | true | | | | |
| 4 | true | | | | |
| 4 | false | | | | |

## Why generative recs stay offline

(p99 on the Qwen listwise path, if you measured it elsewhere, vs HSTU retrieve. Sub-100 ms is the production bar, not a gate on this laptop.)
