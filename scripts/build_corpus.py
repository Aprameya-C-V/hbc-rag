"""Build the HotpotQA and MuSiQue paragraph corpora and load them into Postgres."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from hbc_rag.config import load_env
from hbc_rag.embedding import FastEmbed
from hbc_rag.store.pg import PgVectorStore
from hbc_rag.types import Document

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def hotpot_rows(limit: int, seed: int) -> tuple[list[dict], dict[str, dict]]:
    table = pq.read_table(RAW / "hotpot_val.parquet").to_pylist()
    rng = random.Random(seed)
    picked = rng.sample(table, min(limit, len(table)))
    questions, corpus = [], {}
    for row in picked:
        ctx = row["context"]
        titles = ctx["title"]
        sentences = ctx["sentences"]
        for title, sents in zip(titles, sentences):
            key = f"hotpotqa:{title}"
            corpus.setdefault(key, {"dataset": "hotpotqa", "doc_key": key, "title": title, "body": " ".join(sents)})
        support = sorted({f"hotpotqa:{t}" for t in row["supporting_facts"]["title"]})
        questions.append(
            {
                "qid": f"hotpot:{row['id']}",
                "dataset": "hotpotqa",
                "question": row["question"],
                "answer": row["answer"],
                "support_keys": support,
            }
        )
    return questions, corpus


def musique_rows(limit: int, seed: int) -> tuple[list[dict], dict[str, dict]]:
    rows = []
    with open(RAW / "musique_dev.jsonl") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rng = random.Random(seed)
    rows = [r for r in rows if r.get("answerable")]
    picked = rng.sample(rows, min(limit, len(rows)))
    questions, corpus = [], {}
    for row in picked:
        support = set()
        for para in row["paragraphs"]:
            key = f"musique:{para['title']}"
            corpus.setdefault(
                key,
                {"dataset": "musique", "doc_key": key, "title": para["title"], "body": para["paragraph_text"]},
            )
            if para.get("is_supporting"):
                support.add(key)
        questions.append(
            {
                "qid": f"musique:{row['id']}",
                "dataset": "musique",
                "question": row["question"],
                "answer": row["answer"],
                "support_keys": sorted(support),
            }
        )
    return questions, corpus


async def load(docs: list[dict], batch: int = 256) -> None:
    embedder = FastEmbed()
    async with PgVectorStore() as store:
        before = await store.count_documents()
        for start in range(0, len(docs), batch):
            chunk = docs[start : start + batch]
            vectors = embedder.embed([f"{d['title']}. {d['body']}" for d in chunk])
            records = [Document(**d, embedding=v) for d, v in zip(chunk, vectors)]
            await store.upsert_documents(records)
            print(f"  embedded {min(start + batch, len(docs))}/{len(docs)}")
        after = await store.count_documents()
        print(f"documents: {before} -> {after} (+{after - before})")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpot", type=int, default=300)
    parser.add_argument("--musique", type=int, default=150)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    load_env()
    hp_questions, hp_corpus = hotpot_rows(args.hotpot, args.seed)
    mq_questions, mq_corpus = musique_rows(args.musique, args.seed)
    questions = hp_questions + mq_questions
    corpus = {**hp_corpus, **mq_corpus}

    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    with open(out / "questions.jsonl", "w") as handle:
        for q in questions:
            handle.write(json.dumps(q) + "\n")
    with open(out / "corpus.jsonl", "w") as handle:
        for doc in corpus.values():
            handle.write(json.dumps(doc) + "\n")

    print(f"questions: {len(questions)} (hotpot {len(hp_questions)}, musique {len(mq_questions)})")
    print(f"paragraphs: {len(corpus)} (hotpot {len(hp_corpus)}, musique {len(mq_corpus)})")
    await load(list(corpus.values()))


if __name__ == "__main__":
    asyncio.run(main())
