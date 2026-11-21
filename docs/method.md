# Method

## Question

Does bounding retrieval into fewer, larger hops change answer quality, evidence coverage, cost and tail latency in agentic RAG, and where does the system saturate under offered load?

## Systems under test

| Policy | Chunks per hop | Hop cap | Top-k |
|---|---|---|---|
| `fine` | 1 | 4 | 6 |
| `coarse` | 4 | 2 | 8 |

Both policies share the same retriever, prompt, model, temperature (0) and stopping rule. Only hop granularity differs.

## Data

- HotpotQA distractor validation split, 300 sampled questions, paragraphs deduplicated into one corpus.
- MuSiQue answerable dev split, 50 sampled questions as a held-out check.
- Retrieval unit is the paragraph, so supporting-fact labels map exactly to corpus keys.

Corpus: 5,155 paragraphs across both datasets.

## Retrieval

- Dense: `BAAI/bge-small-en-v1.5`, 384 dimensions, local ONNX embeddings, no API.
- Lexical: Postgres `tsvector` with `ts_rank_cd`.
- Fusion: reciprocal rank fusion in a single SQL statement.
- Index: pgvector HNSW (cosine) plus GIN on the tsvector.
- Namespace: schema `hbc_rag` on Neon Postgres, pooled connection through PgBouncer.

## Model

`deepseek-v4.1-flash` through OpenCode Go at temperature 0. The pipeline asks for JSON: `{"action":"answer"|"retrieve","answer":...,"query":...}`. A retrieve action triggers the next hop with the model's rewritten query.

## Scoring

- Normalized exact match (lowercase, punctuation stripped, articles removed).
- Token F1.
- Supporting-paragraph recall over the union of evidence across hops.
- Hop count, prompt tokens, cached tokens, completion tokens, billed cost from `usage.opencode.allowance`, end-to-end latency.

No LLM judge is used in the headline. That decision is deliberate: deterministic scorers cannot drift between runs, and the judge budget is better spent later on paraphrase cases only.

## Statistics

- Paired by question id.
- Paired bootstrap, 10,000 resamples, 95% percentile intervals, for F1 and cost deltas.
- McNemar on exact-match correctness for paired policy comparison.

## Load protocol

- Open-loop Poisson arrivals, so slow responses do not reduce offered load.
- Latency measured from scheduled arrival, not from send, which avoids coordinated omission.
- SLO 8 seconds. Goodput is the share of requests inside it.
- Runs at 1 and 2 requests per second against the HTTP API.

## Threats to validity

- One seed per dataset. Deltas under 2 points are noise until replicated.
- MuSiQue has 50 questions, so its intervals are wide.
- p95 includes gateway 429 retries, so it measures the served path rather than model latency alone.
- Cache hit rate depends on gateway routing, which varies run to run.
- The corpus is paragraph-level; finer chunking could change absolute scores, though both policies would move together.
