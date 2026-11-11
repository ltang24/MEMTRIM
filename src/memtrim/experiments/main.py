import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from ..baselines import registry as method_registry
from ..core import config as core_config
from ..core import errors, matching
from ..datasets import registry as dataset_registry
from ..evaluation import report
from ..memory import registry as memory_registry


def load_config(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _named(spec):
    if isinstance(spec, str):
        return spec, {}
    (name, params), = spec.items()
    return name, dict(params or {})


class JsonlWriter:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.done = set()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.done.add(self.key(json.loads(line)))

    @staticmethod
    def key(record):
        return record["dataset"], record["item_id"], record["memory_system"], record["method"]

    def is_done(self, dataset, item_id, memory_system, method):
        return (dataset, item_id, memory_system, method) in self.done

    def write(self, record):
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.done.add(self.key(record))


def make_record(item, memory_system, method_name, entries, result):
    extra = {k: v for k, v in result.items() if k not in ("output", "context")}
    return {
        "dataset": item.dataset,
        "item_id": item.id,
        "pair_id": item.pair_id,
        "condition": item.condition,
        "group": item.group,
        "store_id": item.store_id,
        "memory_system": memory_system,
        "method": method_name,
        "retrieved_ids": [e["entry_id"] for e in entries],
        "retrieved_interactions": [e.get("interaction_id") for e in entries],
        "context": result.get("context"),
        "output": result["output"],
        "gold": item.gold,
        "correct": matching.is_correct(result["output"], item.gold, item.aliases, item.answer_type),
        **extra,
    }


def attach_sources(entries, interactions):
    by_id = {ia.interaction_id: ia for ia in interactions}
    for e in entries:
        source = by_id.get(e.get("interaction_id"))
        e["source_text"] = source.text if source is not None else None
    return entries


def run_without_memory(items, methods, llm, writer, workers):
    def work(item):
        for method in methods:
            if not writer.is_done(item.dataset, item.id, "none", method.name):
                writer.write(make_record(item, "none", method.name, [], method.run(item, [], llm)))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(work, items))


def _safe_search(backend, system_name, item, k, interactions):
    try:
        return attach_sources(backend.search(item.query, k), interactions)
    except Exception as e:
        errors.record_failure(system_name, "retrieve", item.id, backend.store_id, e)
        return []


def run_store(dataset, seed, system_name, system_params, store_id, interactions, store_items, methods, llm, writer, top_k):
    pending = [
        (item, m) for item in store_items for m in methods if not writer.is_done(dataset, item.id, system_name, m.name)
    ]
    if not pending:
        return
    backend = memory_registry.create(system_name, f"{dataset}_{seed}_{store_id}", **system_params)
    try:
        for ia in interactions:
            try:
                backend.add(ia.interaction_id, ia.text, ia.answer, item_id=ia.item_id, group=ia.group, is_primary=ia.is_primary)
            except Exception as e:
                errors.record_failure(system_name, "write", ia.item_id, backend.store_id, e)
        bound = [(m, m.for_store(backend, llm)) for m in methods]
        for item in store_items:
            retrieved = {top_k: _safe_search(backend, system_name, item, top_k, interactions)}
            for method, runner in bound:
                if writer.is_done(dataset, item.id, system_name, method.name):
                    continue
                size = method.pool_size
                if size not in retrieved:
                    retrieved[size] = _safe_search(backend, system_name, item, size, interactions)
                entries = retrieved[size]
                writer.write(make_record(item, system_name, method.name, entries, runner.run(item, entries, llm)))
    finally:
        backend.close()


def _select(specs, only):
    named = [_named(s) for s in specs]
    return [(n, p) for n, p in named if only is None or n in only]


def _settings(cfg, args):
    seeds = [args.seed] if args.seed is not None else cfg.get("seeds", [core_config.DEFAULT_SEED])
    n = args.n if args.n is not None else (5 if args.command == "smoke" else cfg.get("n_per_condition", core_config.DEFAULT_N_PER_CONDITION))
    return seeds, n, cfg.get("max_store_size", core_config.DEFAULT_MAX_STORE_SIZE)


def build_data(cfg, args):
    seeds, n, max_store_size = _settings(cfg, args)
    for seed in seeds:
        for name, params in _select(cfg["datasets"], args.datasets):
            items, stores = dataset_registry.build(name, seed=seed, n=n, max_store_size=max_store_size, **params)
            meta = dataset_registry.settings(name, seed, n, max_store_size, params)
            path = dataset_registry.save(name, meta, items, stores)
            print(f"[seed {seed}] {name}: {len(items)} items, {len(stores)} stores -> {path}")


def run(cfg, args):
    seeds, n, max_store_size = _settings(cfg, args)
    top_k = cfg.get("top_k", core_config.DEFAULT_TOP_K)
    workers = cfg.get("workers", core_config.DEFAULT_WORKERS)
    store_workers = cfg.get("store_workers", 1)
    out_dir = os.path.join(core_config.PROJECT_ROOT, cfg.get("output_dir") or core_config.RESULTS_DIR)
    datasets = _select(cfg["datasets"], args.datasets)
    systems = _select(cfg["memory_systems"], args.memories)

    from ..core.llm_client import get_client

    llm = get_client()
    methods = [method_registry.create(name, top_k=top_k, **params) for name, params in _select(cfg["methods"], args.methods)]
    no_memory_methods = [m for m in methods if not m.uses_memory]
    memory_methods = [m for m in methods if m.uses_memory]

    for seed in seeds:
        out_path = os.path.join(out_dir, f"{args.command}_seed{seed}.jsonl")
        writer = JsonlWriter(out_path)
        for ds_name, ds_params in datasets:
            items, stores = dataset_registry.load_or_build(ds_name, seed, n, max_store_size, ds_params)
            print(f"[seed {seed}] {ds_name}: {len(items)} items, {len(stores)} stores")
            run_without_memory(items, no_memory_methods, llm, writer, workers)
            by_store = {}
            for item in items:
                by_store.setdefault(item.store_id, []).append(item)
            for system_name, system_params in systems:
                with ThreadPoolExecutor(max_workers=max(1, store_workers)) as ex:
                    futures = [
                        ex.submit(run_store, ds_name, seed, system_name, system_params, sid, ias, by_store.get(sid, []), memory_methods, llm, writer, top_k)
                        for sid, ias in stores.items()
                    ]
                    for fut in futures:
                        fut.result()
                print(f"[seed {seed}] {ds_name} / {system_name}: done")
        print(f"\nsaved {out_path}\n")
        print(report.render(report.load_predictions(out_path)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="memtrim")
    sub = ap.add_subparsers(dest="command", required=True)
    for command in ("build-data", "smoke", "run"):
        p = sub.add_parser(command)
        p.add_argument("--config", default=os.path.join(core_config.PROJECT_ROOT, "configs", "main.yaml"))
        p.add_argument("--datasets", nargs="+")
        p.add_argument("--memories", nargs="+")
        p.add_argument("--methods", nargs="+")
        p.add_argument("--n", type=int)
        p.add_argument("--seed", type=int)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "build-data":
        build_data(cfg, args)
    else:
        run(cfg, args)


if __name__ == "__main__":
    main()
