"""Measure prefix-cache behaviour on the Go DeepSeek path."""

from __future__ import annotations

import asyncio
import json
import time

from hbc_rag.config import load_env
from hbc_rag.providers.openai_compat import OpenAICompat

FILLER = "Evidence is quoted verbatim and must be cited by title. " * 1600


async def main() -> None:
    load_env()
    generator = OpenAICompat("deepseek-v4.1-flash")
    prefix = "Cache probe session. " + FILLER
    rows = []
    try:
        for i in range(6):
            started = time.perf_counter()
            result = await generator.generate(
                [
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": f"Return the single word ok. Attempt {i}."},
                ],
                max_tokens=24,
            )
            usage = result.raw.get("usage") or {}
            opencode = usage.get("opencode") or {}
            rows.append(
                {
                    "attempt": i,
                    "prompt_tokens": result.usage.prompt_tokens,
                    "cached_tokens": result.usage.cached_tokens,
                    "latency_ms": round(result.latency_ms, 1),
                    "allowance_usd": opencode.get("allowance"),
                    "list_cost_usd": opencode.get("list_cost"),
                }
            )
            print(json.dumps(rows[-1]))
            await asyncio.sleep(5)
    finally:
        await generator.aclose()

    out = __import__("pathlib").Path(__file__).resolve().parents[1] / "reports" / "data"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cache_probe.json").write_text(json.dumps({"rows": rows}, indent=2))

    hits = [r for r in rows if r["cached_tokens"] > 0]
    print(
        json.dumps(
            {
                "attempts": len(rows),
                "cache_hits": len(hits),
                "hit_rate": len(hits) / len(rows),
                "cached_tokens_mean": sum(r["cached_tokens"] for r in rows) / len(rows),
                "allowance_min": min(r["allowance_usd"] or 0 for r in rows),
                "allowance_max": max(r["allowance_usd"] or 0 for r in rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
