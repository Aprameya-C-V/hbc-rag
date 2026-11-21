from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

from .metrics import summarize
from .sut import SystemUnderTest
from .workload import DatasetQuestion


@dataclass
class LoadResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    offered: int = 0
    completed: int = 0
    errors: int = 0
    summary: dict[str, float] = field(default_factory=dict)
    wall_s: float = 0.0


class OpenLoop:
    """Poisson arrivals, fixed rate. Latency measured from arrival, not from send.

    Requests are launched on a schedule independent of completion, so slow
    responses cannot reduce the offered load and inflate the latency estimate.
    """

    def __init__(
        self,
        *,
        sut: SystemUnderTest,
        questions: list[DatasetQuestion],
        rate_per_s: float,
        duration_s: float | None = None,
        concurrency_cap: int = 256,
        slo_ms: float = 6000.0,
        seed: int = 7,
    ):
        self.sut = sut
        self.questions = questions
        self.rate = rate_per_s
        self.duration_s = duration_s or (len(questions) / rate_per_s)
        self.cap = concurrency_cap
        self.slo_ms = slo_ms
        self.rng = random.Random(seed)

    async def run(self) -> LoadResult:
        result = LoadResult()
        started = time.perf_counter()
        semaphore = asyncio.Semaphore(self.cap)
        tasks: list[asyncio.Task] = []
        index = 0
        issue_time = started

        while True:
            now = time.perf_counter()
            if now - started >= self.duration_s:
                break
            if issue_time > now:
                await asyncio.sleep(issue_time - now)
            question = self.questions[index % len(self.questions)]
            index += 1
            result.offered += 1
            task = asyncio.create_task(self._one(question, semaphore, issue_time, result))
            tasks.append(task)
            gap = self.rng.expovariate(self.rate)
            issue_time += gap

        if tasks:
            await asyncio.gather(*tasks)
        result.wall_s = time.perf_counter() - started
        latencies = [r["latency_ms"] for r in result.records if r["ok"]]
        result.summary = summarize(latencies, slo_ms=self.slo_ms)
        result.completed = len(result.records)
        result.errors = sum(1 for r in result.records if not r["ok"])
        return result

    async def _one(
        self,
        question: DatasetQuestion,
        semaphore: asyncio.Semaphore,
        issued_at: float,
        result: LoadResult,
    ) -> None:
        async with semaphore:
            start = time.perf_counter()
            try:
                payload = await self.sut.ask(question.question, deadline_ms=int(self.slo_ms))
                ok = True
                error = ""
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                payload = {}
                ok = False
                error = f"{type(exc).__name__}: {exc}"
            latency = (time.perf_counter() - issued_at) * 1000
            result.records.append(
                {
                    "qid": question.qid,
                    "ok": ok,
                    "error": error,
                    "latency_ms": latency,
                    "service_ms": (time.perf_counter() - start) * 1000,
                    "answer": payload.get("answer", ""),
                    "hops": payload.get("hops", []),
                }
            )
