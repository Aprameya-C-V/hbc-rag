# HBC-RAG

Hop-bounded agentic RAG with concurrency characterization on rented model APIs.

Two independent packages and one app:

| Path | Package | Role |
|---|---|---|
| `packages/rag-core` | `hbc-rag-core` | Retrieval, hop policies, providers, pipeline |
| `packages/harness` | `hbc-harness` | Open-loop load generation, metrics, scoring, statistics |
| `apps/api` | `hbc-api` | FastAPI wrapper, the HTTP system under test |
| `web` | `hbc-rag-web` | Dashboard for health and recorded runs |

The harness knows nothing about Gemini or DeepSeek. The RAG knows nothing about load testing. They meet at one protocol:

```python
class SystemUnderTest(Protocol):
    async def ask(self, question: str, *, deadline_ms: int | None = None) -> dict: ...
```

## Results

DeepSeek V4.1 Flash via OpenCode Go, temperature 0, deterministic scoring, paired by question.

### HotpotQA, 150 questions

| policy | EM | F1 | support recall | hops | p50 | p95 | cost |
|---|---|---|---|---|---|---|---|
| fine (1 chunk/hop) | 0.427 | 0.588 | 0.620 | 1.43 | 4.07s | 28.3s | $0.0160 |
| coarse (4 chunks/hop) | 0.560 | 0.707 | 0.917 | 1.11 | 2.62s | 27.6s | $0.0186 |

Coarse minus fine F1: **+0.119** (95% CI +0.057 to +0.181). McNemar on EM: chi2 13.88, p < 0.001. Cost is $0.000017 higher per question but cost per correct answer is lower ($0.000221 vs $0.000250).

### MuSiQue, 50 held-out questions

| policy | EM | F1 | support recall | hops | p50 | p95 | cost |
|---|---|---|---|---|---|---|---|
| fine | 0.260 | 0.302 | 0.337 | 1.42 | 4.30s | 27.5s | $0.0075 |
| coarse | 0.260 | 0.313 | 0.527 | 1.26 | 4.33s | 12.3s | $0.0095 |

F1 difference +0.011 (95% CI -0.077 to +0.097), so accuracy is a tie at this sample size. Evidence recall and the tail improve.

### Offered load, open loop into the HTTP API, 8s SLO

| load | requests | p50 | p95 | goodput |
|---|---|---|---|---|
| 1 qps | 44 | 2.85s | 26.0s | 84.1% |
| 2 qps | 94 | 10.53s | 76.4s | 47.9% |

The knee sits between 1 and 2 qps, caused by upstream concurrency limits and retry backoff, not by the database. Full report with figures: `reports/report.html`.

### What this says

1. Hop granularity is a real quality lever. Coarse hops raise F1 by 12 points and evidence recall by 30 points on HotpotQA while using fewer hops.
2. The cost mechanism is not prefix caching at these prompt sizes. Cache hit rates were 2 to 9 percent, so savings come from fewer round trips, not cache reads.
3. Tail latency is the binding constraint under load. p95 is dominated by gateway retries, and goodput halves between 1 and 2 qps.

## Backends

| Model | Provider | Cache rule |
|---|---|---|
| `deepseek-v4.1-flash` | OpenCode Go or direct DeepSeek | automatic disk prefix cache, 50x read discount |
| `mimo-v2.6-flash` | OpenCode Go | automatic prefix, 50x discount |
| `hy4-flash-free` | OpenCode Go | free, no cache economics |
| `muse-spark-1.3-contributor` | OpenCode Go | automatic prefix, 50x discount |
| `gemini-3.5-flash-lite` | Gemini Interactions API | implicit only, 4,096 minimum, 2,050-token incremental rule |

Provider note, measured: OpenCode Go does report cache (`prompt_tokens_details.cached_tokens`, 5,376 of 5,441 on a repeated prefix) and bills through `usage.opencode.allowance`, so Go is the default DeepSeek path. The catch is routing: a third identical call returned to a gateway instance without the cached prefix and cost 24 times more. That hit-or-miss cache locality is a result to measure, not a bug to hide. Direct DeepSeek stays available as a control.

## Credentials

Never in this repo.

```
keys.env       model keys, chmod 600, gitignored
.env.local     Neon DATABASE_URL and bucket creds, gitignored
```

`keys.env` in the project root:

```
HBC_GO_API_KEY=
DEEPSEEK_API_KEY=
GEMINI_API_KEY=
HBC_PROVIDER=gemini
HBC_MODEL=gemini-3.5-flash-lite
```

The repo `.env.local` is written by `neon deploy` / `neon env pull` and holds `DATABASE_URL`, `DATABASE_URL_UNPOOLED` and the object storage credentials.

## Setup

```bash
uv sync --all-packages
npm --prefix web install
neon link                 # pins .neon, pulls .env.local
neon deploy               # provisions bucket assets
uv run python scripts/neon_smoke.py
```

Database objects live in schema `hbc_rag`; the extension is installed in `public`.

## Run

```bash
uv run uvicorn app.main:app --app-dir apps/api --port 8000
curl localhost:8000/health
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"Which document mentions gamma rays?","dataset":"smoke"}'
npm run build             # dashboard, served at / by the API
```

Harness:

```bash
uv run hbc-harness run --questions path/to/questions.jsonl --rate 4 --duration 120 --out runs/trace.jsonl
```

## Test

```bash
uv run pytest
npm run build
```

CI runs both on every push.

## Reproduce

```bash
uv run python scripts/build_corpus.py --hotpot 300 --musique 150
uv run python scripts/run_eval.py --dataset hotpotqa --limit 150 --policies fine,coarse --concurrency 4
uv run python scripts/run_eval.py --dataset musique --limit 50 --policies fine,coarse --concurrency 4
uv run hbc-harness run --questions data/hotpotqa_questions.jsonl --rate 1 --duration 30 --slo 8000 --out runs/load-1qps.jsonl
uv run hbc-harness run --questions data/hotpotqa_questions.jsonl --rate 2 --duration 40 --slo 8000 --out runs/load-2qps.jsonl
uv run python scripts/report.py
```

Corpus and run traces stay out of git; the report and figures are committed.

## Status

Complete for this pass. Postgres with pgvector, local embeddings, hybrid retrieval, two hop policies, deterministic scoring with statistics, open-loop load tests, dashboard and generated report. Next: second seed, a larger MuSiQue holdout, and a calibrated judge for paraphrase cases.
