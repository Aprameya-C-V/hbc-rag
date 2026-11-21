from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: int | None = None
    dataset: str
    doc_key: str
    title: str
    body: str
    embedding: list[float] | None = None


class Retrieved(BaseModel):
    id: int
    doc_key: str
    title: str
    body: str
    score: float


class Usage(BaseModel):
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class Hop(BaseModel):
    index: int
    query: str
    retrieved: list[Retrieved]
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    decision: Literal["answer", "retrieve", "stop"] = "answer"
    model_output: str = ""


class RunMeta(BaseModel):
    run_id: str
    model: str
    provider: str
    policy: str
    dataset: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: dict[str, Any] = Field(default_factory=dict)
