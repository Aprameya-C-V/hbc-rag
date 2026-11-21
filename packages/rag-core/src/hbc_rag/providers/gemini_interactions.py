from __future__ import annotations

import time

import httpx

from ..config import gemini_api_key
from ..types import Usage
from .base import Generation

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"


class GeminiInteractions:
    """Gemini Interactions API over REST.

    Stateful when previous_interaction_id is passed. Implicit caching only.
    """

    def __init__(self, model: str = "gemini-3.5-flash-lite", *, name: str = "gemini"):
        self.name = name
        self.model = model
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(180.0, connect=15.0),
            headers={"x-goog-api-key": gemini_api_key(), "content-type": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _text(payload: dict) -> str:
        direct = payload.get("output_text")
        if direct:
            return str(direct).strip()
        parts: list[str] = []
        for step in payload.get("steps") or []:
            if step.get("type") != "model_output":
                continue
            for block in step.get("content") or []:
                if block.get("type") == "text" and block.get("text"):
                    parts.append(str(block["text"]))
        return "\n".join(parts).strip()

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        previous_interaction_id: str | None = None,
    ) -> Generation:
        prompt = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        body = {
            "model": self.model,
            "input": prompt,
            "generation_config": {"temperature": temperature, "max_output_tokens": max_tokens},
        }
        if previous_interaction_id:
            body["previous_interaction_id"] = previous_interaction_id
        started = time.perf_counter()
        response = await self._client.post(BASE_URL, json=body)
        elapsed = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        return Generation(
            text=self._text(payload),
            usage=Usage(
                prompt_tokens=int(usage.get("total_input_tokens") or 0),
                cached_tokens=int(usage.get("total_cached_tokens") or 0),
                completion_tokens=int(usage.get("total_output_tokens") or 0),
                total_tokens=int(usage.get("total_tokens") or 0),
            ),
            latency_ms=elapsed,
            provider=self.name,
            model=self.model,
            raw=payload,
            interaction_id=payload.get("id"),
        )
