from __future__ import annotations

import time
from typing import Any, Protocol

import httpx


class SystemUnderTest(Protocol):
    async def ask(self, question: str, *, deadline_ms: int | None = None) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


class HttpSUT:
    def __init__(self, url: str, *, timeout: float = 120.0):
        self.url = url
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ask(self, question: str, *, deadline_ms: int | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        response = await self._client.post(
            self.url,
            json={"question": question, "deadline_ms": deadline_ms},
        )
        elapsed = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        payload = response.json()
        payload["client_ms"] = elapsed
        return payload
