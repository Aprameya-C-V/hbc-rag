from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..types import Usage


@dataclass
class Generation:
    text: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    provider: str = ""
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    interaction_id: str | None = None


class Generator(Protocol):
    name: str
    model: str

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        previous_interaction_id: str | None = None,
    ) -> Generation: ...
