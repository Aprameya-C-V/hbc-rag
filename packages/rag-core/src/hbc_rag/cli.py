from __future__ import annotations

import argparse
import asyncio
import json

from .config import load_env
from .embedding import DeterministicEmbed, FastEmbed
from .pipeline import RagPipeline
from .policies import CoarseHop, FineHop
from .providers.openai_compat import OpenAICompat
from .store.pg import PgVectorStore


async def _ask(args: argparse.Namespace) -> None:
    load_env()
    store = await PgVectorStore().open()
    try:
        generator = OpenAICompat(args.model)
        policy = CoarseHop() if args.policy == "coarse" else FineHop()
        rag = RagPipeline(store=store, generator=generator, policy=policy, dataset=args.dataset)
        result = await rag.ask(args.question)
        print(json.dumps(
            {
                "answer": result.answer,
                "hops": result.hop_count,
                "usage": result.usage.model_dump(),
                "cost_usd": result.cost_usd,
                "ms": round(result.total_ms, 1),
            },
            indent=2,
        ))
        await generator.aclose()
    finally:
        await store.close()


async def _db_init(_: argparse.Namespace) -> None:
    load_env()
    await PgVectorStore.init_schema()
    print("schema applied")


def main() -> None:
    parser = argparse.ArgumentParser(prog="hbc-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    db = sub.add_parser("db")
    db_sub = db.add_subparsers(dest="db_cmd", required=True)
    db_sub.add_parser("init")

    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--dataset", default="hotpotqa")
    ask.add_argument("--policy", choices=["fine", "coarse"], default="coarse")
    ask.add_argument("--model", default="deepseek-v4.1-flash")

    args = parser.parse_args()
    if args.cmd == "db" and args.db_cmd == "init":
        asyncio.run(_db_init(args))
    elif args.cmd == "ask":
        asyncio.run(_ask(args))
