from __future__ import annotations

import argparse
import asyncio
import json

from .load import OpenLoop
from .sut import HttpSUT
from .trace import write_jsonl
from .workload import load_questions


async def _run(args: argparse.Namespace) -> None:
    questions = load_questions(args.questions)
    sut = HttpSUT(args.url)
    try:
        result = await OpenLoop(
            sut=sut,
            questions=questions,
            rate_per_s=args.rate,
            duration_s=args.duration,
            slo_ms=args.slo,
        ).run()
    finally:
        await sut.aclose()
    if args.out:
        write_jsonl(args.out, result.records)
    print(json.dumps(result.summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="hbc-harness")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--questions", required=True)
    run.add_argument("--url", default="http://127.0.0.1:8000/ask")
    run.add_argument("--rate", type=float, default=1.0)
    run.add_argument("--duration", type=float, default=60.0)
    run.add_argument("--slo", type=float, default=6000.0)
    run.add_argument("--out", default="")
    args = parser.parse_args()
    if args.cmd == "run":
        asyncio.run(_run(args))
