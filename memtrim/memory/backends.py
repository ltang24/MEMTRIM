"""Adapters for Mem0, A-Mem, MemGPT (Letta) and Graphiti, plus a local dense
store used for offline smoke tests.

Each adapter is created per memory store (one store = one retrieval group of
one condition). Heavy imports happen inside constructors so that only the
library you actually run needs to be installed.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..embed import get_encoder
from .base import MemoryBackend, Retrieved, dedupe_by_interaction, interaction_text


# --------------------------------------------------------------- dense (local)
class DenseBackend(MemoryBackend):
    """Plain embedding retrieval over full interactions. Offline / tests only;
    not one of the four systems evaluated in the paper."""

    name = "dense"
    has_timestamps = True

    def __init__(self, store_id: str, cfg: dict[str, Any], llm_cfg: dict[str, Any]) -> None:
        self.enc = get_encoder(cfg.get("encoder", "sentence-transformers/all-mpnet-base-v2"))
        self.texts: list[str] = []
        self.ids: list[str] = []
        self.ts: list[float] = []
        self._mat: np.ndarray | None = None

    def add(self, interaction_id, text, answer, timestamp=None):
        self.texts.append(interaction_text(text, answer) if answer else text)
        self.ids.append(interaction_id)
        self.ts.append(timestamp or 0.0)
        self._mat = None

    def search(self, query, k):
        if not self.texts:
            return []
        if self._mat is None:
            self._mat = self.enc.encode(self.texts)
        sims = self._mat @ self.enc.encode([query])[0]
        order = np.argsort(-sims, kind="stable")
        return dedupe_by_interaction(
            [Retrieved(self.texts[i], self.ids[i], float(sims[i])) for i in order], k
        )


# ----------------------------------------------------------------------- Mem0
class Mem0Backend(MemoryBackend):
    """mem0ai OSS ``Memory``. With ``infer: true`` (default, the system's
    standard mode) an LLM extracts memory entries from each interaction; the
    source interaction id travels in entry metadata."""

    name = "mem0"

    def __init__(self, store_id: str, cfg: dict[str, Any], llm_cfg: dict[str, Any]) -> None:
        from mem0 import Memory

        self.user_id = store_id
        self.infer = bool(cfg.get("infer", True))
        root = Path(cfg.get("path", ".cache/mem0")) / store_id
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        enc = cfg.get("encoder", "sentence-transformers/all-mpnet-base-v2")
        config = {
            "llm": {"provider": "openai", "config": {
                "model": llm_cfg["model"], "temperature": 0.0, "max_tokens": 1500,
                **({"openai_base_url": llm_cfg["base_url"]} if llm_cfg.get("base_url") else {}),
                "api_key": os.environ.get(llm_cfg.get("api_key_env", "OPENAI_API_KEY"), "EMPTY"),
            }},
            "embedder": {"provider": "huggingface", "config": {"model": enc}},
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": "mem", "path": str(root / "qdrant"),
                "embedding_model_dims": int(cfg.get("embedding_dims", 768)), "on_disk": True}},
            "history_db_path": str(root / "history.db"),
        }
        config.update(cfg.get("override", {}))
        self.m = Memory.from_config(config)

    def add(self, interaction_id, text, answer, timestamp=None):
        messages = [{"role": "user", "content": text}]
        if answer:
            messages.append({"role": "assistant", "content": f"Answer: {answer}"})
        self.m.add(messages, user_id=self.user_id, metadata={"interaction_id": interaction_id}, infer=self.infer)

    def search(self, query, k):
        res = self.m.search(query, top_k=max(k * 3, k), filters={"user_id": self.user_id}, threshold=0.0)
        items = res.get("results", res) if isinstance(res, dict) else res
        out = [
            Retrieved(r.get("memory", ""), (r.get("metadata") or {}).get("interaction_id"), float(r.get("score", 0.0)))
            for r in items
        ]
        return dedupe_by_interaction(out, k)


# ---------------------------------------------------------------------- A-Mem
class AMemBackend(MemoryBackend):
    """agiresearch/A-mem ``AgenticMemorySystem``: each interaction becomes a
    note with LLM-generated keywords/context/tags; notes are linked and
    evolved; retrieval uses ``search_agentic`` (similarity + linked
    neighbours). A-Mem uses one global Chroma collection, so stores must be
    processed sequentially (the runner does this)."""

    name = "amem"

    def __init__(self, store_id: str, cfg: dict[str, Any], llm_cfg: dict[str, Any]) -> None:
        from agentic_memory.memory_system import AgenticMemorySystem

        if llm_cfg.get("base_url"):
            os.environ["OPENAI_BASE_URL"] = llm_cfg["base_url"]
        self.sys = AgenticMemorySystem(
            model_name=cfg.get("encoder", "all-MiniLM-L6-v2"),
            llm_backend="openai",
            llm_model=llm_cfg["model"],
            evo_threshold=int(cfg.get("evo_threshold", 100)),
            api_key=os.environ.get(llm_cfg.get("api_key_env", "OPENAI_API_KEY"), "EMPTY"),
        )
        self.note2iid: dict[str, str] = {}

    def add(self, interaction_id, text, answer, timestamp=None):
        content = interaction_text(text, answer) if answer else text
        note_id = self.sys.add_note(content)
        self.note2iid[note_id] = interaction_id

    def search(self, query, k):
        res = self.sys.search_agentic(query, k=k)
        out = []
        for r in res:
            note = self.sys.memories.get(r["id"])
            txt = r.get("content") or (note.content if note else "")
            ctx = r.get("context")
            if ctx and ctx not in ("General", ""):
                txt = f"{txt}\n(Context: {ctx})"
            out.append(Retrieved(txt, self.note2iid.get(r["id"]), float(r.get("score", 0.0) or 0.0)))
        return dedupe_by_interaction(out, k)


# --------------------------------------------------------------- MemGPT/Letta
class MemGPTBackend(MemoryBackend):
    """MemGPT via a running Letta server (``letta server``). Interactions go to
    archival memory; the most recent ``n_core`` interactions are also kept in
    the agent's core (active) memory block, mirroring MemGPT's split between
    in-context and archival memory. Retrieval returns core-memory items first,
    then archival search results, up to k."""

    name = "memgpt"
    has_timestamps = True

    def __init__(self, store_id: str, cfg: dict[str, Any], llm_cfg: dict[str, Any]) -> None:
        from letta_client import Letta

        self.client = Letta(base_url=cfg.get("base_url", "http://localhost:8283"),
                            api_key=os.environ.get(cfg.get("api_key_env", "LETTA_API_KEY"), "EMPTY"))
        self.n_core = int(cfg.get("n_core", 1))
        self.agent = self.client.agents.create(
            name=f"memtrim-{store_id}-{uuid.uuid4().hex[:6]}",
            model=cfg.get("letta_model", llm_cfg["model"]),
            embedding=cfg.get("letta_embedding", "letta/letta-free"),
            memory_blocks=[{"label": "recent_interactions", "value": "", "limit": 20000}],
            include_base_tools=False,
        )
        self.recent: list[tuple[str, str]] = []

    def add(self, interaction_id, text, answer, timestamp=None):
        content = interaction_text(text, answer) if answer else text
        self.client.agents.passages.create(agent_id=self.agent.id, text=content, tags=[f"iid:{interaction_id}"])
        self.recent = (self.recent + [(interaction_id, content)])[-self.n_core:] if self.n_core > 0 else []
        self.client.agents.blocks.update(
            "recent_interactions", agent_id=self.agent.id,
            value="\n\n".join(f"[{i}] {c}" for i, c in self.recent),
        )

    def search(self, query, k):
        out = [Retrieved(c, i, 1.0) for i, c in reversed(self.recent)]
        res = self.client.agents.passages.search(agent_id=self.agent.id, query=query, top_k=k)
        rows = getattr(res, "results", None) or getattr(res, "passages", None) or res
        for r in rows or []:
            r = r if isinstance(r, dict) else getattr(r, "model_dump", lambda: r.__dict__)()
            tags = r.get("tags") or []
            iid = next((t[4:] for t in tags if isinstance(t, str) and t.startswith("iid:")), None)
            out.append(Retrieved(r.get("content") or r.get("text", ""), iid, float(r.get("score", 0.0) or 0.0)))
        return dedupe_by_interaction(out, k)

    def close(self):
        try:
            self.client.agents.delete(self.agent.id)
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------- Graphiti
class _EncoderEmbedder:
    """Graphiti EmbedderClient backed by a local sentence encoder."""

    def __init__(self, name: str) -> None:
        from graphiti_core.embedder.client import EmbedderClient

        enc = get_encoder(name)

        class _E(EmbedderClient):
            async def create(self, input_data):
                text = input_data if isinstance(input_data, str) else " ".join(map(str, input_data))
                return enc.encode([text])[0].tolist()

            async def create_batch(self, input_data_list):
                return [v.tolist() for v in enc.encode(list(input_data_list))]

        self.client = _E()


class GraphitiBackend(MemoryBackend):
    """Graphiti temporal knowledge graph (embedded Kuzu by default; Neo4j via
    ``uri``). Each interaction is an episode; search returns relation facts
    (edges), each linked to the episode(s) it was extracted from."""

    name = "graphiti"
    has_timestamps = True
    _loop: asyncio.AbstractEventLoop | None = None

    def __init__(self, store_id: str, cfg: dict[str, Any], llm_cfg: dict[str, Any]) -> None:
        from graphiti_core import Graphiti
        from graphiti_core.llm_client.config import LLMConfig as GLLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        if GraphitiBackend._loop is None:
            GraphitiBackend._loop = asyncio.new_event_loop()
        self.loop = GraphitiBackend._loop
        self.group = store_id.replace("/", "_").replace(":", "_")
        llm = OpenAIGenericClient(GLLMConfig(
            api_key=os.environ.get(llm_cfg.get("api_key_env", "OPENAI_API_KEY"), "EMPTY"),
            model=llm_cfg["model"], small_model=llm_cfg["model"],
            base_url=llm_cfg.get("base_url"), temperature=0.0,
        ), structured_output_mode=cfg.get("structured_output_mode", "json_object"))
        embedder = _EncoderEmbedder(cfg.get("encoder", "sentence-transformers/all-mpnet-base-v2")).client
        cross = _PassthroughReranker()
        if cfg.get("uri"):
            self.g = Graphiti(cfg["uri"], cfg.get("user", "neo4j"), os.environ.get("NEO4J_PASSWORD", "password"),
                              llm_client=llm, embedder=embedder, cross_encoder=cross)
        else:
            from graphiti_core.driver.kuzu_driver import KuzuDriver

            db = Path(cfg.get("path", ".cache/graphiti")) / f"{self.group}.kuzu"
            if db.exists():
                shutil.rmtree(db) if db.is_dir() else db.unlink()
            db.parent.mkdir(parents=True, exist_ok=True)
            driver = KuzuDriver(db=str(db))
            self.g = Graphiti(graph_driver=driver, llm_client=llm, embedder=embedder, cross_encoder=cross)
            # graphiti-core's KuzuDriver.build_indices_and_constraints is a
            # no-op, so the full-text indices used by search are created here.
            from graphiti_core.driver.driver import GraphProvider
            from graphiti_core.graph_queries import get_fulltext_indices

            for q in get_fulltext_indices(GraphProvider.KUZU):
                self._run(driver.execute_query(q))
        self._run(self.g.build_indices_and_constraints())
        self.ep2iid: dict[str, str] = {}
        self._t = 0

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def add(self, interaction_id, text, answer, timestamp=None):
        from graphiti_core.nodes import EpisodeType

        body = f"user: {text}" + (f"\nassistant: Answer: {answer}" if answer else "")
        self._t += 1
        ref = datetime.fromtimestamp(timestamp or 1_700_000_000 + self._t, tz=timezone.utc)
        res = self._run(self.g.add_episode(
            name=interaction_id, episode_body=body, source=EpisodeType.message,
            source_description="past interaction", reference_time=ref, group_id=self.group,
        ))
        self.ep2iid[res.episode.uuid] = interaction_id

    def search(self, query, k):
        edges = self._run(self.g.search(query, group_ids=[self.group], num_results=max(k * 3, k)))
        out = []
        for e in edges:
            iid = next((self.ep2iid[u] for u in (e.episodes or []) if u in self.ep2iid), None)
            out.append(Retrieved(e.fact, iid, 0.0))
        return dedupe_by_interaction(out, k)

    def close(self):
        try:
            self._run(self.g.close())
        except Exception:  # noqa: BLE001
            pass


class _PassthroughReranker:
    """Keeps Graphiti's own hybrid (BM25 + cosine, RRF) ranking; avoids an
    OpenAI-only cross-encoder dependency."""

    def __new__(cls):
        from graphiti_core.cross_encoder.client import CrossEncoderClient

        class _R(CrossEncoderClient):
            async def rank(self, query, passages):
                n = len(passages)
                return [(p, float(n - i)) for i, p in enumerate(passages)]

        return _R()


BACKENDS = {
    "dense": DenseBackend,
    "mem0": Mem0Backend,
    "amem": AMemBackend,
    "memgpt": MemGPTBackend,
    "graphiti": GraphitiBackend,
}


def make_backend(name: str, store_id: str, backend_cfg: dict[str, Any], llm_cfg: dict[str, Any]) -> MemoryBackend:
    """``backend_cfg`` is the backend's own section of the config (may be {})."""
    return BACKENDS[name](store_id, backend_cfg or {}, llm_cfg)
