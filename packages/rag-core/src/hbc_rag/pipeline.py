from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .config import PRICES
from .embedding import Embedder, FastEmbed
from .policies import HopPolicy
from .providers.base import Generator
from .store.pg import PgVectorStore
from .types import Hop, Usage

SYSTEM = """You are a retrieval agent answering multi-hop questions from evidence.
Return JSON only, with this shape:
{"action":"answer"|"retrieve","answer":"...","query":"...","confidence":0.0}
Use action "retrieve" only when the evidence cannot answer the question and a new search would help.
Keep the answer to a short span, no explanation."""


@dataclass
class RagResult:
    answer: str
    hops: list[Hop] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    total_ms: float = 0.0

    @property
    def hop_count(self) -> int:
        return len(self.hops)

    @property
    def evidence(self) -> list[str]:
        seen: list[str] = []
        for hop in self.hops:
            for item in hop.retrieved:
                if item.doc_key not in seen:
                    seen.append(item.doc_key)
        return seen


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"action": "answer", "answer": text, "confidence": 0.0}


class RagPipeline:
    def __init__(
        self,
        *,
        store: PgVectorStore,
        generator: Generator,
        policy: HopPolicy,
        embedder: Embedder | None = None,
        dataset: str = "hotpotqa",
    ):
        self.store = store
        self.generator = generator
        self.policy = policy
        self.embedder = embedder or FastEmbed()
        self.dataset = dataset

    def _cost(self, usage: Usage) -> float:
        price = PRICES.get(self.generator.model)
        if not price:
            return 0.0
        return price.cost(usage.prompt_tokens, usage.cached_tokens, usage.completion_tokens)

    def _prompt(self, question: str, evidence: list[dict], hop: int) -> list[dict[str, str]]:
        blocks = [
            f"[{item['doc_key']}] {item['title']}\n{item['body']}" for item in evidence
        ]
        context = "\n\n".join(blocks)
        user = (
            f"Question: {question}\n\n"
            f"Evidence gathered so far (hop {hop + 1} of {self.policy.max_hops}):\n{context}\n\n"
            "Answer if the evidence is sufficient, otherwise request one more retrieval."
        )
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    async def ask(self, question: str, *, question_id: str = "") -> RagResult:
        started = time.perf_counter()
        usage = Usage()
        hops: list[Hop] = []
        evidence: list[dict] = []
        query = question
        answer = ""

        for hop_index in range(self.policy.max_hops):
            vector = self.embedder.embed([query])[0]
            retrieved = await self.store.hybrid_search(
                dataset=self.dataset,
                query_text=query,
                query_embedding=vector,
                limit=self.policy.retrieval_k(hop_index),
            )
            for item in retrieved:
                evidence.append(item.model_dump())

            generation = await self.generator.generate(
                self._prompt(question, evidence, hop_index),
                temperature=0.0,
                max_tokens=256,
            )
            usage = usage.add(generation.usage)
            parsed = _extract_json(generation.text)
            decision = parsed.get("action", "answer")
            record = Hop(
                index=hop_index,
                query=query,
                retrieved=retrieved,
                prompt_tokens=generation.usage.prompt_tokens,
                cached_tokens=generation.usage.cached_tokens,
                completion_tokens=generation.usage.completion_tokens,
                latency_ms=generation.latency_ms,
                decision="retrieve" if decision == "retrieve" else "answer",
                model_output=generation.text[:2000],
            )
            hops.append(record)

            if decision != "retrieve":
                answer = str(parsed.get("answer", "")).strip()
                break
            query = str(parsed.get("query") or question)
            answer = str(parsed.get("answer", "")).strip()

        if not answer and hops:
            answer = hops[-1].model_output.strip()

        return RagResult(
            answer=answer,
            hops=hops,
            usage=usage,
            cost_usd=self._cost(usage),
            total_ms=(time.perf_counter() - started) * 1000,
        )
