"""Experiment runner shared by RQ1, Tables 2-4, ablations and runtime.

For every (dataset, memory backend, seed):
  1. Items are grouped into memory stores: one store per (condition, retrieval
     group), split into chunks of at most ``store_size`` interactions.
     The paired stored interaction of every item in the store is written
     (seed-dependent order); other items' memories act as natural distractors.
  2. MEMTRIM parses every stored interaction once at write time (cached) and
     indexes it; queries are parsed once at read time.
  3. For each query the base system retrieves ``pool_k`` items; every method
     sees the same ranked list (top ``top_k`` unless the method re-ranks).
  4. One JSON line per (item, method) is written to
     ``<out>/preds/<dataset>__<backend>__seed<s>.jsonl``. Finished files are
     skipped on re-run, so interrupted runs resume.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from .core import MemTrim
from .embed import Encoder, get_encoder
from .llm import LLM, LLMConfig
from .memory import make_backend
from .memory.base import MemoryBackend, Retrieved
from .methods import METHODS, NEEDS_MEMTRIM, QueryCtx
from .parser import EvidenceParser, dump_record
from .schema import MemoryRecord, MemTrimOptions, ParsedQuery
from .text import is_correct, query_memory_overlap

T0 = 1_700_000_000.0


# ----------------------------------------------------------------- data
def load_items(data_dir: str | Path, dataset: str, limit: int | None = None,
               conditions: Iterable[str] = ("TA", "TD")) -> list[dict]:
    items = [json.loads(l) for l in open(Path(data_dir) / f"{dataset}.jsonl")]
    items = [it for it in items if it["condition"] in set(conditions)]
    if limit:
        by_cond: dict[str, list[dict]] = defaultdict(list)
        for it in items:
            by_cond[it["condition"]].append(it)
        items = [it for c in sorted(by_cond) for it in by_cond[c][:limit]]
    return items


def make_stores(items: list[dict], seed: int, store_size: int) -> list[tuple[str, list[dict]]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for it in items:
        groups[(it["condition"], it["group"])].append(it)
    rng = random.Random(seed)
    stores = []
    for (cond, grp) in sorted(groups):
        members = sorted(groups[(cond, grp)], key=lambda x: x["id"])
        rng.shuffle(members)
        for c in range(0, len(members), store_size):
            chunk = members[c : c + store_size]
            stores.append((f"{cond}-{grp}-{c // store_size}-s{seed}", chunk))
    return stores


# ------------------------------------------------------------------ setup
@dataclass
class Env:
    cfg: dict[str, Any]
    llm: LLM
    parser_llm: LLM
    encoder: Encoder
    out: Path

    @property
    def mem_llm_cfg(self) -> dict[str, Any]:
        return self.cfg.get("memory_llm") or self.cfg["llm"]


def make_env(cfg: dict[str, Any], use_cache: bool = True) -> Env:
    llm = LLM(LLMConfig.from_dict(cfg["llm"]), use_cache=use_cache)
    pcfg = cfg.get("parser_llm") or {**cfg["llm"], "max_tokens": 1500}
    parser_llm = LLM(LLMConfig.from_dict(pcfg), use_cache=True)
    out = Path(cfg.get("output_dir", "results")) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    return Env(cfg, llm, parser_llm, get_encoder(cfg.get("encoder", "sentence-transformers/all-mpnet-base-v2")), out)


class ParseCache:
    """Write-time parse of every stored interaction, read-time parse of every
    query, with one canonical-key registry per dataset."""

    def __init__(self, env: Env, dataset: str) -> None:
        self.parser = EvidenceParser(env.parser_llm, env.encoder, **env.cfg.get("parser", {}))
        self.records: dict[str, MemoryRecord] = {}
        self.queries: dict[str, ParsedQuery] = {}
        self.path = env.out / "parsed" / f"{dataset}.jsonl"

    def memory(self, mem_id: str, text: str, answer: str) -> MemoryRecord:
        if mem_id not in self.records:
            self.records[mem_id] = self.parser.parse_memory(mem_id, text, answer)
        return self.records[mem_id]

    def query(self, key: str, text: str) -> ParsedQuery:
        if key not in self.queries:
            self.queries[key] = self.parser.parse_query(text)
        return self.queries[key]

    def warm(self, items: list[dict]) -> None:
        """Parse memories first, then queries, in a fixed order, so the
        canonical registry is identical across runs."""
        for it in tqdm(sorted(items, key=lambda x: x["id"]), desc="parse memories", leave=False):
            self.memory(it["memory"]["id"], it["memory"]["text"], it["memory"]["answer"])
        for it in tqdm(sorted(items, key=lambda x: x["id"]), desc="parse queries", leave=False):
            self.query(it["id"], it["query"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            for r in self.records.values():
                f.write(dump_record(r) + "\n")
        (self.path.parent / f"{self.path.stem}.parser_stats.json").write_text(json.dumps(self.parser.stats.__dict__, indent=2))


def split_answer(content: str) -> tuple[str, str]:
    """Split a raw record into (x, y) when it ends with an 'Answer:' line."""
    lines = content.rstrip().splitlines()
    if lines and lines[-1].lower().startswith("answer:"):
        return "\n".join(lines[:-1]), lines[-1].split(":", 1)[1].strip()
    return content, ""


def build_store(env: Env, backend_name: str, store_id: str, members: list[dict], seed: int,
                parse: ParseCache | None, options: MemTrimOptions | None,
                extra_writes: list[tuple[str, str]] | None = None) -> tuple[MemoryBackend, MemTrim | None, dict]:
    mcfg = env.cfg["memory"]
    backend = make_backend(backend_name, store_id, mcfg.get(backend_name, {}), env.mem_llm_cfg)
    mt = MemTrim(options or MemTrimOptions(), encode=env.encoder.encode) if parse else None
    rng = random.Random(seed)
    order = list(members)
    rng.shuffle(order)
    lookup: dict[str, tuple[str, str]] = {}
    for i, it in enumerate(order):
        m = it["memory"]
        ts = T0 + i
        backend.add(m["id"], m["text"], m["answer"], timestamp=ts)
        lookup[m["id"]] = (m["text"], m["answer"])
        if mt is not None:
            rec = parse.memory(m["id"], m["text"], m["answer"])
            rec.timestamp = ts if backend.has_timestamps else None
            mt.add_memory(rec)
    for j, (iid, content) in enumerate(extra_writes or []):
        backend.add_raw(iid, content, timestamp=T0 + len(order) + j)
        x, y = split_answer(content)
        lookup[iid] = (x, y)
        if mt is not None:
            rec = parse.memory(iid, x, y)
            mt.add_memory(rec)
    return backend, mt, lookup


def run_query(env: Env, item: dict, backend: MemoryBackend, mt: MemTrim | None, lookup: dict,
              parse: ParseCache | None, methods: list[str], query_text: str | None = None,
              gold: tuple[str, list[str]] | None = None, timing: bool = False) -> list[dict]:
    """Run every method on one query; returns result rows."""
    mcfg = env.cfg["memory"]
    k, pool_k = int(mcfg.get("top_k", 5)), int(mcfg.get("pool_k", 15))
    q_item = dict(item)
    if query_text is not None:
        q_item["query"] = query_text
    g_ans, g_alias = gold if gold else (item["gold"], item.get("aliases", []))

    t_ret = time.perf_counter()
    retrieved: list[Retrieved] = backend.search(q_item["query"], pool_k)
    t_ret = time.perf_counter() - t_ret

    top = retrieved[:k]
    top_iid = top[0].interaction_id if top else None
    overlap = (
        query_memory_overlap(q_item["query"], lookup[top_iid][0], lookup[top_iid][1])
        if top_iid in lookup else None
    )
    paired_rank = next((r.rank for r in retrieved if r.interaction_id == item["memory"]["id"]), None)

    rows = []
    for name in methods:
        t = time.perf_counter()
        pq = None
        if name in NEEDS_MEMTRIM:
            key = item["id"] if query_text is None else f"{item['id']}::{hash(query_text)}"
            if timing:  # read-time parse is part of MEMTRIM's inference cost
                with parse.parser.llm.no_cache():
                    pq = parse.parser.parse_query(q_item["query"])
            else:
                pq = parse.query(key, q_item["query"])
        ctx = QueryCtx(item=q_item, retrieved=retrieved, k=k, lookup=lookup, llm=env.llm,
                       encoder=env.encoder, memtrim=mt, parser=parse.parser if parse else None,
                       parsed_query=pq, params=env.cfg.get("params", {}))
        pred = METHODS[name](ctx)
        dt = time.perf_counter() - t + (t_ret if name != "no_memory" else 0.0)
        rows.append(dict(
            id=item["id"], pair_id=item["pair_id"], dataset=item["dataset"], condition=item["condition"],
            method=name, prediction=pred, gold=g_ans,
            correct=bool(is_correct(pred, g_ans, g_alias, item["answer_type"])),
            retrieved=[r.interaction_id for r in top], paired_rank=paired_rank,
            overlap_top1=overlap, seconds=dt,
        ))
    return rows


def run_main(cfg: dict[str, Any]) -> Path:
    env = make_env(cfg, use_cache=not cfg.get("timing", False))
    methods = cfg["methods"]
    needs_parse = any(m in NEEDS_MEMTRIM for m in methods)
    for dataset in cfg["datasets"]:
        items = load_items(cfg.get("data_dir", "data/processed"), dataset, cfg.get("limit"))
        parse = ParseCache(env, dataset) if needs_parse else None
        if parse:
            parse.warm(items)
        for backend_name in cfg["memory"]["backends"]:
            for seed in cfg["seeds"]:
                env.llm.config.seed = seed
                path = env.out / "preds" / f"{dataset}__{backend_name}__seed{seed}.jsonl"
                done = path.with_suffix(".done")
                if done.exists():
                    print(f"skip {path.name} (done)")
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w") as f:
                    stores = make_stores(items, seed, int(cfg["memory"].get("store_size", 20)))
                    for store_id, members in tqdm(stores, desc=f"{dataset}/{backend_name}/s{seed}"):
                        backend, mt, lookup = build_store(env, backend_name, store_id, members, seed, parse, None)
                        try:
                            for it in members:
                                for row in run_query(env, it, backend, mt, lookup, parse, methods,
                                                     timing=cfg.get("timing", False)):
                                    row.update(backend=backend_name, seed=seed, store=store_id)
                                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                                f.flush()
                        finally:
                            backend.close()
                done.write_text("ok")
    return env.out
