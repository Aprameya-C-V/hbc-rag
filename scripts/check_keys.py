"""Check which model credentials are live, without printing them."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from hbc_rag.config import KEYS_FILE, gemini_api_key, go_api_key, load_env

TIMEOUT = 20


def check(label: str, url: str, headers: dict[str, str]) -> None:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = response.read(200)
            print(f"{label:10} OK       {response.status} {body[:80]!r}")
    except urllib.error.HTTPError as exc:
        detail = exc.read(160).decode(errors="replace").replace("\n", " ")
        print(f"{label:10} INVALID  {exc.code} {detail}")
    except Exception as exc:  # noqa: BLE001
        print(f"{label:10} ERROR    {type(exc).__name__} {exc}")


def main() -> None:
    load_env()
    print(f"keys file: {KEYS_FILE} exists={KEYS_FILE.exists()}")

    key = go_api_key()
    check(
        "go",
        "https://opencode.ai/inference/go/openai/v1/models",
        {"Authorization": f"Bearer {key}", "User-Agent": "hbc-rag/0.1"},
    )

    deepseek = os.getenv("DEEPSEEK_API_KEY")
    if deepseek:
        check("deepseek", "https://api.deepseek.com/models", {"Authorization": f"Bearer {deepseek}"})
    else:
        print("deepseek   MISSING  DEEPSEEK_API_KEY is empty")

    try:
        gemini = gemini_api_key()
    except RuntimeError:
        gemini = ""
    if gemini:
        check(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/models",
            {"x-goog-api-key": gemini},
        )
    else:
        print("gemini     MISSING  no GEMINI_API_KEY and no ~/.config/gemini-api-key")


if __name__ == "__main__":
    main()
