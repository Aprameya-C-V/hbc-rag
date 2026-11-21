"""Prove Postgres (pgvector) and Object Storage are wired end to end."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone

import boto3

from hbc_rag.config import load_env
from hbc_rag.embedding import DeterministicEmbed
from hbc_rag.store.pg import PgVectorStore
from hbc_rag.types import Document, RunMeta

FIXTURES = [
    ("doc-alpha", "Alpha", "The first fixture document about alpha particles."),
    ("doc-beta", "Beta", "The second fixture document about beta decay."),
    ("doc-gamma", "Gamma", "The third fixture document about gamma rays."),
    ("doc-delta", "Delta", "The fourth fixture document about delta waves."),
    ("doc-epsilon", "Epsilon", "The fifth fixture document about epsilon values."),
]


def s3_roundtrip() -> dict:
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"],
        region_name=os.environ.get("AWS_REGION", "ap-southeast-1"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    bucket = os.getenv("HBC_BUCKET", "assets")
    key = f"smoke/{uuid.uuid4().hex}.json"
    body = json.dumps({"run": "smoke", "ts": time.time()}).encode()
    started = time.perf_counter()
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    fetched = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    client.delete_object(Bucket=bucket, Key=key)
    return {
        "bucket": bucket,
        "bytes": len(body),
        "identical": fetched == body,
        "ms": round((time.perf_counter() - started) * 1000, 1),
    }


async def main() -> None:
    load_env()
    print("schema:", end=" ")
    await PgVectorStore.init_schema()
    print("applied")

    embedder = DeterministicEmbed()
    docs = [
        Document(
            dataset="smoke",
            doc_key=key,
            title=title,
            body=body,
            embedding=embedder.embed([f"{title} {body}"])[0],
        )
        for key, title, body in FIXTURES
    ]
    async with PgVectorStore() as store:
        written = await store.upsert_documents(docs)
        total = await store.count_documents("smoke")
        print(f"documents: wrote={written} total_in_smoke={total}")

        source = docs[2]
        hits = await store.hybrid_search(
            dataset="smoke",
            query_text=f"{source.title} {source.body}",
            query_embedding=source.embedding or [],
            limit=3,
        )
        print("vector search:", [(h.doc_key, round(h.score, 4)) for h in hits])
        assert hits and hits[0].doc_key == "doc-gamma", "nearest neighbour is not the query document"

        run_id = f"smoke-{uuid.uuid4().hex[:8]}"
        await store.insert_run(
            RunMeta(
                run_id=run_id,
                model="none",
                provider="smoke",
                policy="none",
                dataset="smoke",
                started_at=datetime.now(timezone.utc),
                tags={"source": "neon_smoke"},
            ),
            {"documents": total, "top_hit": hits[0].doc_key},
        )
        runs = await store.list_runs(5)
        print("runs:", [(r["id"], r["metrics"]) for r in runs[:2]])
        assert any(r["id"] == run_id for r in runs)

    storage = await asyncio.to_thread(s3_roundtrip)
    print("object storage:", storage)
    assert storage["identical"], "object roundtrip mismatch"

    print("PASS postgres+pgvector+object-storage")


if __name__ == "__main__":
    asyncio.run(main())
