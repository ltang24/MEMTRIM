import argparse
import os
import time

from . import config, errors, metrics, pipeline
from .llm_client import get_client
from .memtrim import AblationFlags


def _add_common_args(p):
    p.add_argument("--n", type=int, default=config.DEFAULT_N_PER_CONDITION)
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    p.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    p.add_argument("--max-store-size", type=int, default=config.DEFAULT_MAX_STORE_SIZE)
    p.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS)
    p.add_argument("--keep-shared", action="store_true")
    p.add_argument("--keep-conflicts", action="store_true")
    p.add_argument("--keep-old-answer", action="store_true")
    p.add_argument("--ignore-task-change", action="store_true")
    p.add_argument("--embedding-match", action="store_true")


def _flags_from_args(args):
    return AblationFlags(
        keep_shared=args.keep_shared,
        keep_conflicts=args.keep_conflicts,
        keep_old_answer=args.keep_old_answer,
        ignore_task_change=args.ignore_task_change,
        embedding_match=args.embedding_match,
    )


def cmd_build_data(args):
    items, stores = pipeline.build_dataset(args.n, args.seed, args.max_store_size)
    items_path = os.path.join(config.DATA_DIR, "items.json")
    stores_path = os.path.join(config.DATA_DIR, "stores.json")
    pipeline.save_dataset(items, stores, items_path, stores_path)
    print(f"items: {len(items)} ({len(items) // 2} edits x 2 conditions)")
    print(f"stores: {len(stores)}")
    print(f"saved {items_path}")
    print(f"saved {stores_path}")


def _print_error_counts():
    counts = errors.counts()
    print("\n=== write/retrieve error counts ===")
    if not counts:
        print("none")
        return
    for (system_name, stage), n in sorted(counts.items()):
        print(f"{system_name:<6} {stage:<10} {n}")


def cmd_smoke(args):
    errors.reset()
    client = get_client()
    items, stores = pipeline.build_dataset(args.n, args.seed, args.max_store_size)
    print(f"built {len(items)} items across {len(stores)} stores")

    mem0_stores, amem_stores = pipeline.build_memory_systems(stores)
    tries, parsed = pipeline.index_stores(client, mem0_stores, amem_stores, args.workers)

    print("\n=== sample parsed stored memories ===")
    shown = 0
    for store_id in parsed["mem0"]:
        for frag_id, record in parsed["mem0"][store_id].items():
            print(f"-- mem0 fragment {frag_id} --")
            print("source:", record.source_text)
            print("parse_ok:", record.parse_ok)
            for u in record.evidence:
                print("  evidence:", u.entity, "|", u.attribute, "|", u.value, "| multi_valued=", u.multi_valued, "| span=", repr(u.span))
            print("  task:", record.task)
            print("  outcome_text:", record.outcome_text)
            shown += 1
            if shown >= 3:
                break
        if shown >= 3:
            break

    print("\n=== sample parsed queries ===")
    shown = 0
    for item in items:
        query_text = f"{item.current_fact} {item.question}"
        record = pipeline.parser.parse_text(client, query_text)
        print(f"-- query for {item.item_id} --")
        print("text:", query_text)
        print("parse_ok:", record.parse_ok)
        for u in record.evidence:
            print("  evidence:", u.entity, "|", u.attribute, "|", u.value)
        print("  task:", record.task)
        shown += 1
        if shown >= 3:
            break

    flags = _flags_from_args(args)
    predictions = pipeline.run_answering(client, items, mem0_stores, amem_stores, tries, parsed, flags, args.top_k, args.workers)

    print("\n=== base vs memtrim memory blocks for two TD_DUP items ===")
    shown = 0
    seen_items = set()
    for p in predictions:
        if p["condition"] != "TD_DUP" or p["method"] not in ("base", "memtrim"):
            continue
        key = (p["item_id"], p["memory_system"])
        if p["item_id"] in seen_items:
            continue
        pair = [q for q in predictions if q["item_id"] == p["item_id"] and q["memory_system"] == p["memory_system"] and q["method"] in ("base", "memtrim")]
        if len(pair) < 2:
            continue
        seen_items.add(p["item_id"])
        print(f"-- item {p['item_id']} / system {p['memory_system']} --")
        for q in pair:
            print(f"[{q['method']}] block:")
            print(q["memory_block"])
            print(f"[{q['method']}] output: {q['raw_output']!r} correct={q['correct']}")
        shown += 1
        if shown >= 2:
            break

    print("\n=== summary ===")
    rows = metrics.summarize(predictions)
    metrics.print_summary(rows)
    _print_error_counts()

    results_path = os.path.join(config.RESULTS_DIR, "smoke_predictions.jsonl")
    pipeline.save_predictions(predictions, results_path)
    print(f"\nsaved {results_path}")


def cmd_run(args):
    errors.reset()
    client = get_client()
    t0 = time.time()
    items, stores = pipeline.build_dataset(args.n, args.seed, args.max_store_size)
    items_path = os.path.join(config.DATA_DIR, "items.json")
    stores_path = os.path.join(config.DATA_DIR, "stores.json")
    pipeline.save_dataset(items, stores, items_path, stores_path)
    print(f"built {len(items)} items across {len(stores)} stores in {time.time() - t0:.1f}s")

    t1 = time.time()
    mem0_stores, amem_stores = pipeline.build_memory_systems(stores)
    print(f"built memory systems in {time.time() - t1:.1f}s")

    t2 = time.time()
    tries, parsed = pipeline.index_stores(client, mem0_stores, amem_stores, args.workers)
    print(f"indexed stores in {time.time() - t2:.1f}s")

    flags = _flags_from_args(args)
    t3 = time.time()
    predictions = pipeline.run_answering(client, items, mem0_stores, amem_stores, tries, parsed, flags, args.top_k, args.workers)
    print(f"answered {len(predictions)} predictions in {time.time() - t3:.1f}s")

    results_path = os.path.join(config.RESULTS_DIR, f"predictions_n{args.n}_seed{args.seed}.jsonl")
    pipeline.save_predictions(predictions, results_path)
    print(f"saved {results_path}")

    rows = metrics.summarize(predictions)
    metrics.print_summary(rows)
    _print_error_counts()
    print(f"\ntotal runtime: {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(prog="memtrim")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-data")
    _add_common_args(p_build)
    p_build.set_defaults(func=cmd_build_data)

    p_smoke = sub.add_parser("smoke")
    _add_common_args(p_smoke)
    p_smoke.set_defaults(func=cmd_smoke)

    p_run = sub.add_parser("run")
    _add_common_args(p_run)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
