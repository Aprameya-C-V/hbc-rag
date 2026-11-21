from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbed:
    """Local ONNX embeddings, no API and no quota."""

    dim = 384

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [np.asarray(v, dtype=np.float32).tolist() for v in self._model.embed(texts)]


class DeterministicEmbed:
    """Stable hash embedding for tests and offline smoke checks."""

    dim = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            vector = rng.normal(size=self.dim).astype(np.float32)
            vector /= np.linalg.norm(vector)
            out.append(vector.tolist())
        return out
