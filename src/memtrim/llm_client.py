import hashlib
import json
import os
import sqlite3
import threading
import time

from openai import OpenAI
from transformers import AutoTokenizer

from . import config

_thread_local = threading.local()
_tokenizer = None
_tokenizer_lock = threading.Lock()


def get_tokenizer():
    global _tokenizer
    with _tokenizer_lock:
        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(config.MEMTRIM_MODEL_HF_NAME)
        return _tokenizer


def count_prompt_tokens(messages):
    tok = get_tokenizer()
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return len(ids)


class PromptTooLongError(Exception):
    pass


def _cache_connection():
    conn = getattr(_thread_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(config.CACHE_DB_PATH, timeout=60)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS calls ("
            "cache_key TEXT PRIMARY KEY, "
            "model TEXT, "
            "request_json TEXT, "
            "response_json TEXT, "
            "created_at REAL)"
        )
        conn.commit()
        _thread_local.conn = conn
    return conn


def _cache_key(model, messages, params):
    payload = json.dumps(
        {"model": model, "messages": messages, "params": params},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VLLMClient:
    def __init__(self, base_url=None, model=None, max_model_len=None):
        self.base_url = base_url or config.MEMTRIM_BASE_URL
        self.model = model or config.MEMTRIM_MODEL
        self.client = OpenAI(base_url=self.base_url, api_key="sk-memtrim")
        self.max_model_len = max_model_len or self._fetch_max_model_len()

    def _fetch_max_model_len(self):
        models = self.client.models.list()
        served = None
        for entry in models.data:
            if entry.id == self.model:
                served = entry
                break
        if served is None:
            raise RuntimeError(f"model {self.model!r} not found at {self.base_url}, server reports {[m.id for m in models.data]}")
        max_len = getattr(served, "max_model_len", None)
        print(f"vLLM server at {self.base_url} is serving model={served.id} root={served.root} max_model_len={max_len}")
        if max_len is None:
            raise RuntimeError("server did not report max_model_len")
        return max_len

    def chat(self, messages, response_format=None, max_tokens=1024, temperature=0.0, seed=0, extra_body=None):
        n_tokens = count_prompt_tokens(messages)
        budget = self.max_model_len - max_tokens
        if n_tokens > budget:
            raise PromptTooLongError(
                f"prompt has {n_tokens} tokens, generation budget needs {max_tokens}, "
                f"server max_model_len={self.max_model_len} leaves only {budget} tokens for the prompt"
            )
        params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        if response_format is not None:
            params["response_format"] = response_format
        if extra_body:
            params["extra_body"] = extra_body
        cache_key = _cache_key(self.model, messages, params)
        conn = _cache_connection()
        row = conn.execute(
            "SELECT response_json FROM calls WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is not None:
            return json.loads(row[0])
        kwargs = dict(params)
        request_extra_body = kwargs.pop("extra_body", None)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body=request_extra_body,
            **kwargs,
        )
        content = response.choices[0].message.content
        result = {"content": content}
        conn.execute(
            "INSERT OR REPLACE INTO calls (cache_key, model, request_json, response_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                cache_key,
                self.model,
                json.dumps({"messages": messages, "params": params}, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
                time.time(),
            ),
        )
        conn.commit()
        return result


_default_client = None
_default_client_lock = threading.Lock()


def get_client():
    global _default_client
    with _default_client_lock:
        if _default_client is None:
            _default_client = VLLMClient()
        return _default_client
