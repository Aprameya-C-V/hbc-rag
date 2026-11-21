from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from ..config import database_url, database_url_unpooled
from ..types import Document, Retrieved, RunMeta

SCHEMA = Path(__file__).with_name("schema.sql").read_text()


def vec_literal(values: Iterable[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


class PgVectorStore:
    def __init__(self, dsn: str | None = None, *, min_size: int = 1, max_size: int = 8):
        self.dsn = dsn or database_url()
        self._pool = AsyncConnectionPool(
            conninfo=self.dsn,
            min_size=min_size,
            max_size=max_size,
            open=False,
            kwargs={"row_factory": dict_row, "prepare_threshold": None},
        )

    async def open(self) -> "PgVectorStore":
        await self._pool.open(wait=True)
        return self

    async def close(self) -> None:
        await self._pool.close()

    async def __aenter__(self) -> "PgVectorStore":
        return await self.open()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @staticmethod
    async def init_schema(dsn: str | None = None) -> None:
        pool = AsyncConnectionPool(
            conninfo=dsn or database_url_unpooled(),
            min_size=1,
            max_size=2,
            open=False,
            kwargs={"prepare_threshold": None},
        )
        await pool.open(wait=True)
        try:
            async with pool.connection() as conn:
                await conn.execute(SCHEMA)
        finally:
            await pool.close()

    async def upsert_documents(self, docs: list[Document]) -> int:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    """
                    insert into documents (dataset, doc_key, title, body, embedding)
                    values (%s, %s, %s, %s, %s::vector)
                    on conflict (doc_key) do update
                      set title = excluded.title,
                          body = excluded.body,
                          embedding = excluded.embedding
                    """,
                    [
                        (d.dataset, d.doc_key, d.title, d.body, vec_literal(d.embedding or []))
                        for d in docs
                    ],
                )
        return len(docs)

    async def hybrid_search(
        self,
        *,
        dataset: str,
        query_text: str,
        query_embedding: list[float],
        limit: int = 10,
        candidates: int = 40,
    ) -> list[Retrieved]:
        sql = """
        with dense as (
          select id, row_number() over (order by embedding <=> %(vec)s::vector) as r
          from documents where dataset = %(dataset)s
          order by embedding <=> %(vec)s::vector limit %(candidates)s
        ), lex as (
          select id, row_number() over (
                   order by ts_rank_cd(tsv, plainto_tsquery('english', %(q)s)) desc) as r
          from documents
          where dataset = %(dataset)s and tsv @@ plainto_tsquery('english', %(q)s)
          limit %(candidates)s
        ), fused as (
          select coalesce(d.id, l.id) as id,
                 coalesce(1.0 / (60 + d.r), 0) + coalesce(1.0 / (60 + l.r), 0) as score
          from dense d full outer join lex l on d.id = l.id
        )
        select doc.id, doc.doc_key, doc.title, doc.body, f.score
        from fused f join documents doc on doc.id = f.id
        order by f.score desc limit %(limit)s
        """
        params: dict[str, Any] = {
            "vec": vec_literal(query_embedding),
            "q": query_text,
            "dataset": dataset,
            "candidates": candidates,
            "limit": limit,
        }
        async with self._pool.connection() as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
        return [
            Retrieved(id=r["id"], doc_key=r["doc_key"], title=r["title"], body=r["body"], score=float(r["score"]))
            for r in rows
        ]

    async def get_document(self, doc_id: int) -> dict[str, Any] | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "select id, dataset, doc_key, title, body, embedding::text as embedding from documents where id = %s",
                (doc_id,),
            )
            row = await cur.fetchone()
        if not row:
            return None
        row = dict(row)
        row["embedding"] = [float(x) for x in row["embedding"].strip("[]").split(",")]
        return row

    async def count_documents(self, dataset: str | None = None) -> int:
        async with self._pool.connection() as conn:
            if dataset:
                cur = await conn.execute("select count(*) as n from documents where dataset = %s", (dataset,))
            else:
                cur = await conn.execute("select count(*) as n from documents")
            row = await cur.fetchone()
        return int(row["n"])

    async def insert_run(self, meta: RunMeta, metrics: dict[str, Any] | None = None) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                insert into runs (id, model, provider, policy, dataset, created_at, metrics, tags)
                values (%s, %s, %s, %s, %s, %s, %s, %s)
                on conflict (id) do update set metrics = excluded.metrics
                """,
                (
                    meta.run_id,
                    meta.model,
                    meta.provider,
                    meta.policy,
                    meta.dataset,
                    meta.started_at,
                    psycopg_json(metrics or {}),
                    psycopg_json(meta.tags),
                ),
            )

    async def insert_hops(self, run_id: str, question_id: str, hops: list[dict[str, Any]]) -> int:
        if not hops:
            return 0
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    """
                    insert into hops (run_id, question_id, hop_index, query, retrieved,
                                      prompt_tokens, cached_tokens, completion_tokens, latency_ms, decision)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            run_id,
                            question_id,
                            hop["index"],
                            hop["query"],
                            psycopg_json(hop.get("retrieved", [])),
                            hop.get("prompt_tokens", 0),
                            hop.get("cached_tokens", 0),
                            hop.get("completion_tokens", 0),
                            hop.get("latency_ms", 0.0),
                            hop.get("decision", "answer"),
                        )
                        for hop in hops
                    ],
                )
        return len(hops)

    async def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "select id, model, provider, policy, dataset, created_at, metrics from runs order by created_at desc limit %s",
                (limit,),
            )
            rows = await cur.fetchall()
        return [dict(r) for r in rows]


def psycopg_json(value: Any) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(value)
