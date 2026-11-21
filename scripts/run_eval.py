"""Run a policy over a dataset and persist per-question results and metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import uuid
from pathlib import Path

import numpy as np

from hbc_harness.eval import exact_match, f1, support_recall
from hbc_rag.config import load_env
from hbc_rag.embedding import FastEmbed
from hbc_rag.pipeline import RagPipeline
from hbc_rag.policies import CoarseHop, FineHop
from hbc_rag.providers.factory import make_generator
from hbc_rag.store.pg import PgVectorStore
from hbc_rag.types import RunMeta

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {"fine": FineHop, "coarse": CoarseHop}


def load_questions(dataset: str, limit: int, seed: int) -> list[dict]:
    rows = []
    with open(ROOT / "data" / "questions.jsonl") as handle:
        for line in handle:
            row = json.loads(line)
            if row["dataset"] == dataset:
                rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit]


async def run_policy(
    *,
    store: PgVectorStore,
    generator,
    embedder,
    questions: list[dict],
    policy_name: str,
    dataset: str,
    concurrency: int,
    seed: int,
) -> Path:
    policy = POLICIES[policy_name]()
    rag = RagPipeline(store=store, generator=generator, policy=policy, embedder=embedder, dataset=dataset)
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    done = 0
    started = time.perf_counter()

    async def one(question: dict) -> None:
        nonlocal done
        async with semaphore:
            try:
                result = await rag.ask(question["question"], question_id=question["qid"])
                row = {
                    "qid": question["qid"],
                    "question": question["question"],
                    "gold": question["answer"],
                    "answer": result.answer,
                    "em": exact_match(result.answer, question["answer"]),
                    "f1": f1(result.answer, question["answer"]),
                    "support_recall": support_recall(result.evidence, question["support_keys"]),
                    "hops": result.hop_count,
                    "prompt_tokens": result.usage.prompt_tokens,
                    "cached_tokens": result.usage.cached_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "cost_usd": result.cost_usd,
                    "latency_ms": result.total_ms,
                    "evidence": result.evidence,
                    "error": "",
                }
            except Exception as exc:  # noqa: BLE001
                row = {
                    "qid": question["qid"],
                    "question": question["question"],
                    "gold": question["answer"],
                    "answer": "",
                    "em": False,
                    "f1": 0.0,
                    "support_recall": 0.0,
                    "hops": 0,
                    "prompt_tokens": 0,
                    "cached_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_ms": 0.0,
                    "evidence": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(row)
            done += 1
            if done % 25 == 0:
                print(f"  {policy_name}: {done}/{len(questions)}")

    await asyncio.gather(*(one(q) for q in questions))
    wall = time.perf_counter() - started

    run_id = f"{dataset}-{policy_name}-{uuid.uuid4().hex[:6]}"
    out = ROOT / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w") as handle:
        for row in results:
            handle.write(json.dumps(row, default=str) + "\n")

    latencies = [r["latency_ms"] for r in results if not r["error"]]
    prompt = sum(r["prompt_tokens"] for r in results)
    cached = sum(r["cached_tokens"] for r in results)
    metrics = {
        "questions": len(results),
        "errors": sum(1 for r in results if r["error"]),
        "em": float(np.mean([r["em"] for r in results])),
        "f1": float(np.mean([r["f1"] for r in results])),
        "support_recall": float(np.mean([r["support_recall"] for r in results])),
        "hops_mean": float(np.mean([r["hops"] for r in results])),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "cache_hit_rate": (cached / prompt) if prompt else 0.0,
        "completion_tokens": sum(r["completion_tokens"] for r in results),
        "cost_usd": sum(r["cost_usd"] for r in results),
        "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "wall_s": wall,
    }
    with open(out / "summary.json", "w") as handle:
        json.dump({"run_id": run_id, "policy": policy_name, "dataset": dataset, **metrics}, handle, indent=2)

    await store.insert_run(
        RunMeta(
            run_id=run_id,
            model=generator.model,
            provider=generator.name,
            policy=policy_name,
            dataset=dataset,
            tags={"questions": len(results), "seed": seed},
        ),
        metrics,
    )
    print(json.dumps({"run_id": run_id, **metrics}, indent=2))
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "musique"])
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--policies", default="fine,coarse")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    load_env()
    questions = load_questions(args.dataset, args.limit, args.seed)
    print(f"{len(questions)} questions from {args.dataset}")
    embedder = FastEmbed()
    generator = make_generator()
    store = await PgVectorStore().open()
    try:
        for name in [p.strip() for p in args.policies.split(",") if p.strip()]:
            await run_policy(
                store=store,
                generator=generator,
                embedder=embedder,
                questions=questions,
                policy_name=name,
                dataset=args.dataset,
                concurrency=args.concurrency,
                seed=args.seed,
            )
    finally:
        await store.close()
        if hasattr(generator, "aclose"):
            await generator.aclose()


if __name__ == "__main__":
    asyncio.run(main())
