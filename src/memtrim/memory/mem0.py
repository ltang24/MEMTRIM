import json
import os
import shutil
from types import SimpleNamespace

from ..core import config, errors
from .base import MemoryBackend, make_entry

os.environ.setdefault("MEM0_TELEMETRY", "False")

_EMBEDDING_DIMS = 768

_original_generate_response = None


def _normalize_memory_list_json(text):
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    if not isinstance(data, dict) or not isinstance(data.get("memory"), list):
        return text
    normalized = [{"text": item} if isinstance(item, str) else item for item in data["memory"]]
    if normalized == data["memory"]:
        return text
    data["memory"] = normalized
    return json.dumps(data)


def _generate_response_with_normalized_memory_list(self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs):
    result = _original_generate_response(self, messages, response_format=response_format, tools=tools, tool_choice=tool_choice, **kwargs)
    if not isinstance(result, str):
        return result
    return _normalize_memory_list_json(result)


def _patch_mem0():
    global _original_generate_response
    from mem0.llms.openai import OpenAILLM

    if _original_generate_response is None:
        _original_generate_response = OpenAILLM.generate_response
        OpenAILLM.generate_response = _generate_response_with_normalized_memory_list


def _mem0_config(collection_name, path):
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": config.MEMTRIM_MODEL,
                "openai_base_url": config.MEMTRIM_BASE_URL,
                "api_key": config.FAKE_API_KEY,
                "temperature": 0.0,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": config.ENCODER_MODEL_NAME,
                "model_kwargs": {"device": config.MEMTRIM_ENCODER_DEVICE},
                "embedding_dims": _EMBEDDING_DIMS,
            },
        },
        "vector_store": {
            "provider": "faiss",
            "config": {
                "path": path,
                "collection_name": collection_name,
                "embedding_model_dims": _EMBEDDING_DIMS,
            },
        },
        "version": "v1.1",
    }


def _add_with_retry(memory, store_id, interaction):
    last_error = None
    for _ in range(2):
        try:
            return memory.add(
                [{"role": "user", "content": interaction.text}],
                user_id=store_id,
                metadata={
                    "interaction_id": interaction.interaction_id,
                    "item_id": interaction.item_id,
                    "is_primary": interaction.is_primary,
                },
            )
        except Exception as e:
            last_error = e
    errors.record_failure("mem0", "write", interaction.item_id, store_id, last_error)
    return None


def open_store(store_id):
    _patch_mem0()
    from mem0 import Memory

    path = os.path.join(config.MEM0_STORE_DIR, store_id)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return Memory.from_config(_mem0_config(store_id, path))


def add_interaction(memory, store_id, interaction):
    result = _add_with_retry(memory, store_id, interaction)
    if result is None:
        return []
    return [
        {
            "id": entry["id"],
            "text": entry["memory"],
            "interaction_id": interaction.interaction_id,
            "item_id": interaction.item_id,
        }
        for entry in result.get("results", [])
    ]


def retrieve(memory, store_id, query_text, top_k, item_id=None):
    last_error = None
    for _ in range(2):
        try:
            result = memory.search(query_text, filters={"user_id": store_id}, top_k=top_k)
            entries = []
            for r in result.get("results", []):
                metadata = r.get("metadata") or {}
                entries.append(
                    {
                        "id": r["id"],
                        "text": r["memory"],
                        "interaction_id": metadata.get("interaction_id"),
                        "item_id": metadata.get("item_id"),
                        "score": r.get("score"),
                    }
                )
            return entries
        except Exception as e:
            last_error = e
    errors.record_failure("mem0", "retrieve", item_id, store_id, last_error)
    return []


class Mem0Backend(MemoryBackend):
    name = "mem0"

    def __init__(self, store_id, **kwargs):
        super().__init__(store_id)
        self.memory = open_store(store_id)
        self.fragments = []

    def add(self, interaction_id, text, answer, item_id=None, group=None, is_primary=True):
        interaction = SimpleNamespace(
            interaction_id=interaction_id, text=self.content(text, answer), item_id=item_id, is_primary=is_primary, relation=group
        )
        self.fragments.extend(add_interaction(self.memory, self.store_id, interaction))

    def search(self, query, k):
        hits = retrieve(self.memory, self.store_id, query, k)
        return [make_entry(h["id"], h["text"], h["interaction_id"], h["score"]) for h in hits]

    def entries(self):
        return [make_entry(f["id"], f["text"], f["interaction_id"]) for f in self.fragments]

    def close(self):
        self.memory = None
