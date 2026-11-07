"""Unified chat-LLM client with an on-disk cache and timing.

Backends
--------
* ``openai``  - any OpenAI-compatible endpoint. Use this for GPT models
  (``OPENAI_API_KEY``) and for open-weight models served by vLLM
  (``base_url: http://localhost:8000/v1``).
* ``mock``    - deterministic, offline rule-based backend used only by the
  unit tests and ``--smoke`` runs. It is NOT a model and its outputs carry
  no experimental meaning.

All calls use deterministic decoding by default (temperature 0). The cache key
covers the model, messages and decoding parameters, so re-running an
experiment with the same config reproduces the same outputs and does not
re-query the API.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LLMConfig:
    backend: str = "openai"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int | None = 0
    cache_dir: str | None = ".cache/llm"
    timeout: float = 120.0
    max_retries: int = 5
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LLMConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class _Cache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self._db.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
        return None if row is None else row[0]

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (key, value))
            self._db.commit()


class LLM:
    """Thin chat wrapper. ``chat`` returns the assistant text."""

    def __init__(self, config: LLMConfig, use_cache: bool = True) -> None:
        self.config = config
        self.use_cache = use_cache and config.cache_dir is not None
        self._cache = (
            _Cache(Path(config.cache_dir) / f"{_slug(config.model)}.sqlite")
            if self.use_cache
            else None
        )
        self._client = None
        self.n_calls = 0
        self.n_api_calls = 0
        self.api_seconds = 0.0
        if config.backend == "openai":
            from openai import OpenAI

            self._client = OpenAI(
                base_url=config.base_url,
                api_key=os.environ.get(config.api_key_env, "EMPTY"),
                timeout=config.timeout,
                max_retries=0,
            )
        elif config.backend != "mock":
            raise ValueError(f"Unknown LLM backend: {config.backend}")

    # ------------------------------------------------------------------ API
    @contextlib.contextmanager
    def no_cache(self):
        """Temporarily bypass the cache (used for wall-clock timing)."""
        saved, self._cache = self._cache, None
        try:
            yield self
        finally:
            self._cache = saved

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        json_mode: bool = False,
        tag: str = "",
    ) -> str:
        temperature = self.config.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.config.max_tokens
        seed = self.config.seed if seed is None else seed
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "json": json_mode,
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self.n_calls += 1
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                return hit
        start = time.perf_counter()
        if self.config.backend == "mock":
            out = _mock_response(messages, json_mode)
        else:
            out = self._call_openai(messages, temperature, max_tokens, seed, json_mode)
        self.api_seconds += time.perf_counter() - start
        self.n_api_calls += 1
        if self._cache is not None:
            self._cache.put(key, out)
        return out

    def _call_openai(self, messages, temperature, max_tokens, seed, json_mode) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if seed is not None:
            kwargs["seed"] = seed
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body
        delay = 2.0
        for attempt in range(self.config.max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - retry on any transport error
                if attempt == self.config.max_retries - 1:
                    raise
                # Some servers reject response_format/seed; drop and retry.
                msg = str(exc).lower()
                if "response_format" in msg:
                    kwargs.pop("response_format", None)
                if "seed" in msg:
                    kwargs.pop("seed", None)
                time.sleep(delay)
                delay = min(delay * 2, 60)
        return ""


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in ``text`` (tolerates code fences)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("No JSON found in model output")


# ---------------------------------------------------------------------------
# Offline mock backend (tests / smoke runs only)
# ---------------------------------------------------------------------------
_MOCK_TAG = re.compile(r"<<MOCK:(\w+)>>")


def _mock_response(messages: list[dict[str, str]], json_mode: bool) -> str:
    """Deterministic stand-in. Prompts embed a ``<<MOCK:kind>>`` tag so the mock
    knows which output shape to emit. Answers are heuristic, not meaningful."""
    from .mock import mock_dispatch

    text = "\n".join(m["content"] for m in messages)
    m = _MOCK_TAG.search(text)
    kind = m.group(1) if m else "answer"
    return mock_dispatch(kind, text)
