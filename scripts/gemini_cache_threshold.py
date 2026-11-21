"""Probe the Gemini Interactions incremental-cache threshold.

For each delta size, one stateful chain: a large stable prefix, then a second
turn that adds N tokens. Recorded cached tokens on turn two show whether the
prior request was reused.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from hbc_rag.config import gemini_api_key, load_env

BASE = "https://generativelanguage.googleapis.com/v1beta/interactions"
FILLER = "Retrieval evidence is quoted verbatim. " * 900


async def call(client: httpx.AsyncClient, model: str, payload: dict) -> dict:
    response = await client.post(BASE, json=payload)
    response.raise_for_status()
    return response.json()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-4-hydrogen")
    parser.add_argument("--deltas", default="200,800,1500,2100,3000")
    args = parser.parse_args()

    load_env()
    deltas = [int(d) for d in args.deltas.split(",")]
    rows = []
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(240.0), headers={"x-goog-api-key": gemini_api_key()}
    ) as client:
        for index, delta in enumerate(deltas):
            prefix = f"Trial {index} nonce {index * 7919}. " + FILLER
            first = await call(
                client,
                args.model,
                {"model": args.model, "input": f"{prefix}\nQuestion: name the colour of the sky."},
            )
            added = ("filler " * delta).strip()
            second = await call(
                client,
                args.model,
                {
                    "model": args.model,
                    "input": f"{added}\nQuestion: name a number between one and ten.",
                    "previous_interaction_id": first["id"],
                },
            )
            usage = second.get("usage") or {}
            row = {
                "delta_tokens_requested": delta,
                "prompt_tokens": usage.get("total_input_tokens"),
                "cached_tokens": usage.get("total_cached_tokens"),
                "cache_rate": (
                    (usage.get("total_cached_tokens") or 0) / (usage.get("total_input_tokens") or 1)
                ),
                "latency_hint": len(added),
            }
            rows.append(row)
            print(json.dumps(row))
            await asyncio.sleep(2)

    out = Path(__file__).resolve().parents[1] / "reports" / "data"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"model": args.model, "rows": rows}
    (out / "gemini_threshold.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
