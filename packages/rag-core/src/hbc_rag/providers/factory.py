from __future__ import annotations

import os

from .base import Generator
from .gemini_interactions import GeminiInteractions
from .openai_compat import OpenAICompat


def make_generator(model: str | None = None, *, provider: str | None = None) -> Generator:
    """Resolve a generator from model slug and provider preference.

    provider: gemini | go | deepseek. Defaults to HBC_PROVIDER, then "gemini".
    """
    provider = (provider or os.getenv("HBC_PROVIDER") or "").lower() or None
    model = model or os.getenv("HBC_MODEL") or ""

    if provider is None:
        provider = "gemini" if (not model or model.startswith("gemini")) else "go"

    if provider == "gemini":
        return GeminiInteractions(model or "gemini-3.5-flash-lite")
    if provider == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        return OpenAICompat(
            model or "deepseek-flash",
            base_url="https://api.deepseek.com",
            api_key=key,
            name="deepseek",
        )
    return OpenAICompat(model or "deepseek-v4.1-flash")
