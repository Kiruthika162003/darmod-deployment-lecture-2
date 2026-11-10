"""The cheapest model call is the one you never make.

Two hundred people ask a support bot the same question in a day. Answering
it two hundred times costs two hundred times as much as answering it once
and remembering the answer.

Caching a model response is a dictionary lookup on a hash of everything
that could change the answer: the prompt, the model, and the sampling
settings. If any of those differ the answer could differ, so they all go
into the key.

Which is exactly why prompt templating matters, and why it is on the same
slide. Put a timestamp in your prompt and every request hashes differently
and nothing ever hits. Keep the wording fixed and let only the user's
question vary, and the same question asked twice hits. The `templating`
command below shows both keys side by side.

Calls Gemini through Vertex when GOOGLE_CLOUD_PROJECT is set, or through
the Gemini API with GOOGLE_API_KEY. Cache lives in Redis if REDIS_URL is
set, otherwise a local JSON file, so it runs with neither.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

CACHE_FILE = Path(__file__).parent / ".cache" / "responses.json"
MODEL = "gemini-2.5-flash"

# Published rate for Gemini 2.5 Flash output tokens, used to put a number
# on what a repeat call would have cost. An estimate, and labelled as one.
USD_PER_MILLION_OUTPUT_TOKENS = 2.50

SYSTEM_PROMPT = (
    "You are an assistant for the Vector Institute. "
    "Answer in no more than four sentences."
)


def cache_key(prompt: str, model: str, temperature: float) -> str:
    """Everything that could change the answer goes into the key."""
    material = json.dumps(
        {"prompt": prompt, "model": model, "temperature": temperature},
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class Cache:
    """Redis when REDIS_URL is set, otherwise a JSON file next to this script."""

    def __init__(self) -> None:
        self._redis = None
        url = os.environ.get("REDIS_URL")
        if url and url.lower() not in {"off", "none", ""}:
            try:
                import redis

                self._redis = redis.from_url(url, socket_connect_timeout=2)
                self._redis.ping()
            except Exception as exc:
                print(f"[Redis unavailable ({exc.__class__.__name__}), using a file]")
                self._redis = None

    @property
    def backend(self) -> str:
        return "redis" if self._redis else f"file {CACHE_FILE.name}"

    def _load(self) -> dict[str, str]:
        if not CACHE_FILE.exists():
            return {}
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    def get(self, key: str) -> str | None:
        if self._redis:
            value = self._redis.get(key)
            return value.decode("utf-8") if value else None
        return self._load().get(key)

    def set(self, key: str, value: str) -> None:
        if self._redis:
            self._redis.set(key, value)
            return
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        entries = self._load()
        entries[key] = value
        CACHE_FILE.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def clear(self) -> None:
        if self._redis:
            self._redis.flushdb()
        CACHE_FILE.unlink(missing_ok=True)


def build_prompt(question: str) -> str:
    """The templated prompt. Fixed wording, only the question varies."""
    return f"{SYSTEM_PROMPT}\n\nQuestion: {question}"


def call_gemini(prompt: str, temperature: float) -> tuple[str, int]:
    """Return the answer and how many output tokens it used."""
    from google import genai
    from google.genai import types

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project:
        client = genai.Client(
            vertexai=True,
            project=project,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        )
    elif os.environ.get("GOOGLE_API_KEY"):
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    else:
        raise RuntimeError(
            "Set GOOGLE_CLOUD_PROJECT (Vertex) or GOOGLE_API_KEY (Gemini API)."
        )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    usage = getattr(response, "usage_metadata", None)
    tokens = getattr(usage, "candidates_token_count", 0) or 0
    return response.text, int(tokens)


def ask(question: str, temperature: float) -> int:
    cache = Cache()
    prompt = build_prompt(question)
    key = cache_key(prompt, MODEL, temperature)

    started = time.perf_counter()
    cached = cache.get(key)
    if cached is not None:
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(f"HIT  {elapsed_ms:.1f} ms, $0.00, from {cache.backend}")
        print(f"key  {key[:16]}...\n")
        print(cached)
        return 0

    answer, tokens = call_gemini(prompt, temperature)
    elapsed_ms = (time.perf_counter() - started) * 1000
    cost = tokens / 1_000_000 * USD_PER_MILLION_OUTPUT_TOKENS

    print(f"MISS {elapsed_ms:.1f} ms, {tokens} output tokens, about ${cost:.6f}")
    print(f"key  {key[:16]}... (stored in {cache.backend})\n")
    print(answer)

    cache.set(key, answer)
    return 0


def show_templating(question: str) -> int:
    """Two prompts, same question, one of which can never be cached."""
    volatile = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Current time: {datetime.now(UTC).isoformat()}\n"
        f"Question: {question}"
    )
    templated = build_prompt(question)

    print("\nWith a timestamp in the prompt:")
    print(f"  key now  {cache_key(volatile, MODEL, 0.0)[:32]}")
    time.sleep(1.05)
    volatile_later = volatile.replace(
        volatile.split("Current time: ")[1].split("\n")[0],
        datetime.now(UTC).isoformat(),
    )
    print(f"  key +1s  {cache_key(volatile_later, MODEL, 0.0)[:32]}")
    print("  Different key every second, so the cache never hits.\n")

    print("Templated, only the question varies:")
    print(f"  key now  {cache_key(templated, MODEL, 0.0)[:32]}")
    print(f"  key +1s  {cache_key(build_prompt(question), MODEL, 0.0)[:32]}")
    print("  Same key, so the second person asking pays nothing.\n")
    print("Add the time after the cache lookup, never inside the prompt.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    ask_cmd = sub.add_parser("ask", help="ask a question, cached")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--temperature", type=float, default=0.0)

    sub.add_parser("clear", help="empty the cache")

    templating = sub.add_parser("templating", help="show why templating matters")
    templating.add_argument("question", nargs="?", default="What is Vector Institute?")

    args = parser.parse_args(argv)

    if args.command == "clear":
        Cache().clear()
        print("Cache cleared.")
        return 0
    if args.command == "templating":
        return show_templating(args.question)
    return ask(args.question, args.temperature)


if __name__ == "__main__":
    sys.exit(main())
