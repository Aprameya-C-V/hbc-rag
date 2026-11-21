from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _find_project_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".env.local").exists() or (candidate / "neon.ts").exists():
            return candidate
    return here


ROOT = _find_project_root()


KEYS_FILE = ROOT / "keys.env"
FALLBACK_KEYS_FILE = Path.home() / ".config" / "hbc-rag" / "keys.env"


def load_env() -> None:
    for name in (".env.local", ".env"):
        path = ROOT / name
        if path.exists():
            load_dotenv(path, override=False)
            break
    for keys in (KEYS_FILE, FALLBACK_KEYS_FILE):
        if keys.exists():
            load_dotenv(keys, override=False)


@dataclass(frozen=True)
class ModelPrice:
    input_usd_per_m: float
    cached_usd_per_m: float
    output_usd_per_m: float

    def cost(self, prompt: int, cached: int, completion: int) -> float:
        fresh = max(prompt - cached, 0)
        return (
            fresh * self.input_usd_per_m
            + cached * self.cached_usd_per_m
            + completion * self.output_usd_per_m
        ) / 1_000_000


PRICES: dict[str, ModelPrice] = {
    "deepseek-v4.1-flash": ModelPrice(0.15, 0.003, 0.60),
    "mimo-v2.6-flash": ModelPrice(0.14, 0.0028, 0.28),
    "hy4-flash-free": ModelPrice(0.0, 0.0, 0.0),
    "muse-spark-1.3-contributor": ModelPrice(0.10, 0.002, 0.20),
    "gemini-3.5-flash-lite": ModelPrice(0.0, 0.0, 0.0),
}


def go_base_url() -> str:
    return os.getenv("HBC_GO_BASE_URL", "https://opencode.ai/inference/go/openai/v1")


def go_api_key() -> str:
    key = os.getenv("HBC_GO_API_KEY")
    if key:
        return key
    auth = Path.home() / ".local" / "share" / "opencode" / "auth.json"
    if auth.exists():
        import json

        data = json.loads(auth.read_text())
        key = (data.get("opencode-go") or {}).get("key")
        if key:
            return key
    raise RuntimeError("HBC_GO_API_KEY is not set")


def gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        path = Path.home() / ".config" / "gemini-api-key"
        if path.exists():
            key = path.read_text().strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return key


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return url


def database_url_unpooled() -> str:
    return os.getenv("DATABASE_URL_UNPOOLED") or database_url()
