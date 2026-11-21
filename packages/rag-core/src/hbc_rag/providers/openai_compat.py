from __future__ import annotations

import time

import httpx

from ..config import go_api_key, go_base_url
from ..types import Usage
from .base import Generation


def _usage_from(payload: dict) -> Usage:
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = (
        details.get("cached_tokens")
        or usage.get("prompt_cache_hit_tokens")
        or usage.get("cache_read_input_tokens")
        or 0
    )
    return Usage(
        prompt_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        cached_tokens=int(cached or 0),
        completion_tokens=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
    )


class OpenAICompat:
    """Any OpenAI-compatible chat-completions endpoint.

    Defaults to OpenCode Go, which expects an `api-key` header in addition to
    the standard bearer token.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        name: str = "opencode-go",
        timeout: float = 120.0,
    ):
        self.model = model
        self.name = name
        self.base_url = (base_url or go_base_url()).rstrip("/")
        self.api_key = api_key or go_api_key()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=15.0),
            headers={
                "content-type": "application/json",
                "api-key": self.api_key,
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "hbc-rag/0.1",
                "x-opencode-session": "hbc-rag-p1",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        previous_interaction_id: str | None = None,
    ) -> Generation:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        started = time.perf_counter()
        response = await self._client.post(f"{self.base_url}/chat/completions", json=body)
        elapsed = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        text = ""
        if choices:
            text = ((choices[0].get("message") or {}).get("content") or "").strip()
        opencode = ((payload.get("usage") or {}).get("opencode") or {})
        billing = float(opencode.get("allowance") or opencode.get("list_cost") or 0.0)
        return Generation(
            text=text,
            usage=_usage_from(payload),
            latency_ms=elapsed,
            provider=self.name,
            model=self.model,
            raw=payload,
            billing_usd=billing,
        )
