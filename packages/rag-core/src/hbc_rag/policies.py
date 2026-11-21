from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HopPolicy:
    name: str
    chunks_per_hop: int
    max_hops: int
    top_k: int

    def retrieval_k(self, hop: int) -> int:
        return min(self.top_k, self.chunks_per_hop * (hop + 1))


@dataclass(frozen=True)
class FineHop(HopPolicy):
    name: str = "fine"
    chunks_per_hop: int = 1
    max_hops: int = 4
    top_k: int = 6


@dataclass(frozen=True)
class CoarseHop(HopPolicy):
    name: str = "coarse"
    chunks_per_hop: int = 4
    max_hops: int = 2
    top_k: int = 8
