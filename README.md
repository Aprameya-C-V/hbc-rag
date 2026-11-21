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

## Backends

| Model | Provider | Cache rule |
|---|---|---|
| `deepseek-v4.1-flash` | OpenCode Go or direct DeepSeek | automatic disk prefix cache, 50x read discount |
| `mimo-v2.6-flash` | OpenCode Go | automatic prefix, 50x discount |
| `hy4-flash-free` | OpenCode Go | free, no cache economics |
| `muse-spark-1.3-contributor` | OpenCode Go | automatic prefix, 50x discount |
| `gemini-3.5-flash-lite` | Gemini Interactions API | implicit only, 4,096 minimum, 2,050-token incremental rule |

Provider note, measured: the OpenCode Go gateway reports `cached_tokens: 0` for DeepSeek even on identical 5.4k-token prefixes, while the direct DeepSeek API reports 5,248 cached. Run cache experiments on direct DeepSeek and Gemini, not through the gateway.

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

## Status

P1 wiring is live: Neon Postgres with pgvector and hybrid retrieval, object storage roundtrip, Gemini Interactions end-to-end, runs and hops persisted. Corpus build and the load grid are next.
