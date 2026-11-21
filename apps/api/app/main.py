from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import boto3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hbc_rag.config import ROOT, load_env
from hbc_rag.pipeline import RagPipeline
from hbc_rag.policies import CoarseHop
from hbc_rag.providers.openai_compat import OpenAICompat
from hbc_rag.store.pg import PgVectorStore
from hbc_rag.types import RunMeta

load_env()

BUCKET = os.getenv("HBC_BUCKET", "assets")


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
        region_name=os.getenv("AWS_REGION", "ap-southeast-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = await PgVectorStore().open()
    app.state.s3 = s3_client()
    app.state.embedder = None
    app.state.generator = None
    yield
    await app.state.store.close()
    if app.state.generator is not None:
        await app.state.generator.aclose()


app = FastAPI(title="hbc-rag api", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    dataset: str = "hotpotqa"
    deadline_ms: int | None = None


def _s3_roundtrip(client) -> dict[str, Any]:
    key = f"health/{uuid.uuid4().hex}.json"
    body = json.dumps({"ok": True, "ts": time.time()}).encode()
    started = time.perf_counter()
    client.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType="application/json")
    fetched = client.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    client.delete_object(Bucket=BUCKET, Key=key)
    elapsed = (time.perf_counter() - started) * 1000
    return {"ok": fetched == body, "bucket": BUCKET, "roundtrip_ms": round(elapsed, 1)}


@app.get("/health")
async def health() -> dict[str, Any]:
    store: PgVectorStore = app.state.store
    started = time.perf_counter()
    documents = await store.count_documents()
    db_ms = (time.perf_counter() - started) * 1000
    s3 = await asyncio.to_thread(_s3_roundtrip, app.state.s3)
    return {
        "status": "ok" if s3["ok"] else "degraded",
        "database": {"ok": True, "documents": documents, "latency_ms": round(db_ms, 1)},
        "object_storage": s3,
    }


@app.get("/api/runs")
async def runs(limit: int = 50) -> dict[str, Any]:
    store: PgVectorStore = app.state.store
    rows = await store.list_runs(limit)
    return {"runs": rows, "count": len(rows)}


@app.get("/api/documents/{doc_id}/similar")
async def similar(doc_id: int, limit: int = 5) -> dict[str, Any]:
    store: PgVectorStore = app.state.store
    doc = await store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")
    hits = await store.hybrid_search(
        dataset=doc["dataset"],
        query_text=f"{doc['title']} {doc['body'][:400]}",
        query_embedding=doc["embedding"],
        limit=limit,
    )
    return {"source": {"id": doc["id"], "title": doc["title"]}, "hits": [h.model_dump() for h in hits]}


@app.post("/ask")
async def ask(request: AskRequest) -> dict[str, Any]:
    if app.state.embedder is None:
        from hbc_rag.embedding import FastEmbed

        app.state.embedder = await asyncio.to_thread(FastEmbed)
    if app.state.generator is None:
        provider = os.getenv("HBC_PROVIDER", "gemini")
        if provider == "gemini":
            from hbc_rag.providers.gemini_interactions import GeminiInteractions

            app.state.generator = GeminiInteractions(os.getenv("HBC_MODEL", "gemini-3.5-flash-lite"))
        else:
            app.state.generator = OpenAICompat(os.getenv("HBC_MODEL", "deepseek-v4.1-flash"))

    rag = RagPipeline(
        store=app.state.store,
        generator=app.state.generator,
        policy=CoarseHop(),
        embedder=app.state.embedder,
        dataset=request.dataset,
    )
    result = await rag.ask(request.question)
    run_id = f"ask-{uuid.uuid4().hex[:8]}"
    await app.state.store.insert_run(
        RunMeta(
            run_id=run_id,
            model=app.state.generator.model,
            provider=app.state.generator.name,
            policy=rag.policy.name,
            dataset=request.dataset,
            tags={"route": "/ask"},
        ),
        {
            "hops": result.hop_count,
            "answer": result.answer,
            "prompt_tokens": result.usage.prompt_tokens,
            "cached_tokens": result.usage.cached_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "cost_usd": result.cost_usd,
            "total_ms": round(result.total_ms, 1),
        },
    )
    await app.state.store.insert_hops(run_id, request.question, [h.model_dump() for h in result.hops])
    return {
        "run_id": run_id,
        "answer": result.answer,
        "hops": [h.model_dump() for h in result.hops],
        "evidence": result.evidence,
        "usage": result.usage.model_dump(),
        "cost_usd": result.cost_usd,
        "total_ms": round(result.total_ms, 1),
    }


dist = ROOT / "web" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="web")
