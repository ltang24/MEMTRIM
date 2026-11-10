import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import alias, amem_adapter, config, data_ripple, encoder, equivalence, mem0_adapter, memtrim, parser, prompts, trie as trie_mod
from .llm_client import get_client


def build_dataset(n, seed, max_store_size):
    items = data_ripple.sample_items(n=n, seed=seed)
    stores = data_ripple.build_stores(items, max_store_size=max_store_size)
    return items, stores


def build_memory_systems(stores):
    mem0_stores = {}
    amem_stores = {}
    for store_id, interactions in stores.items():
        memory, fragments = mem0_adapter.build_store(store_id, interactions)
        mem0_stores[store_id] = {"memory": memory, "fragments": fragments}
        system, id_map, fragments2 = amem_adapter.build_store(store_id, interactions)
        amem_stores[store_id] = {"system": system, "id_map": id_map, "fragments": fragments2}
    return mem0_stores, amem_stores


def index_stores(client, mem0_stores, amem_stores, workers):
    encode_fn = encoder.encode
    key_fn = equivalence.make_key_equiv_fn(client)
    value_fn = equivalence.make_value_equiv_fn(client)
    task_fn = equivalence.make_task_equiv_fn(client)

    tries = {"mem0": {}, "amem": {}}
    parsed = {"mem0": {}, "amem": {}}

    def new_trie():
        return trie_mod.Trie(embed_fn=encode_fn, equiv_key_fn=key_fn, equiv_value_fn=value_fn, equiv_task_fn=task_fn)

    tasks = []
    for store_id, entry in mem0_stores.items():
        tries["mem0"][store_id] = new_trie()
        parsed["mem0"][store_id] = {}
        for frag in entry["fragments"]:
            tasks.append(("mem0", store_id, frag))
    for store_id, entry in amem_stores.items():
        tries["amem"][store_id] = new_trie()
        parsed["amem"][store_id] = {}
        for frag in entry["fragments"]:
            tasks.append(("amem", store_id, frag))

    def work(task):
        system_name, store_id, frag = task
        record = parser.parse_text(client, frag["text"])
        return system_name, store_id, frag["id"], record

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, t) for t in tasks]
        for fut in as_completed(futures):
            system_name, store_id, frag_id, record = fut.result()
            parsed[system_name][store_id][frag_id] = record

    for system_name, stores_dict in (("mem0", mem0_stores), ("amem", amem_stores)):
        for store_id, entry in stores_dict.items():
            trie = tries[system_name][store_id]
            for frag in entry["fragments"]:
                record = parsed[system_name][store_id][frag["id"]]
                for i, unit in enumerate(record.evidence):
                    trie.insert(unit, frag["id"], record.support_mask[i])
                if record.outcome_evidence is not None:
                    trie.insert(record.outcome_evidence, frag["id"], False)
                if record.task is not None:
                    trie.insert_task(record.task)

    return tries, parsed


def _answer(client, current_fact, question, memory_block, gold_forms):
    messages = prompts.build_answer_messages(current_fact, question, memory_block)
    response = client.chat(messages, max_tokens=64, temperature=0.0, seed=0)
    text = response["content"]
    correct = alias.contains_gold(text, gold_forms)
    return text, correct


def process_item(client, item, mem0_stores, amem_stores, tries, parsed, no_memory_cache, no_memory_lock, flags, top_k):
    predictions = []
    query_text = f"{item.current_fact} {item.question}"

    with no_memory_lock:
        cached = no_memory_cache.get(item.edit_id)
    if cached is None:
        text, correct = _answer(client, item.current_fact, item.question, None, item.new_target_forms)
        cached = {"text": text, "correct": correct}
        with no_memory_lock:
            no_memory_cache[item.edit_id] = cached
    predictions.append(
        {
            "item_id": item.item_id,
            "edit_id": item.edit_id,
            "condition": item.condition,
            "relation": item.relation,
            "method": "no_memory",
            "memory_system": "none",
            "retrieved_ids": [],
            "memory_block": None,
            "raw_output": cached["text"],
            "correct": cached["correct"],
        }
    )

    query_record = parser.parse_text(client, query_text)

    systems = (
        (
            "mem0",
            lambda: mem0_adapter.retrieve(
                mem0_stores[item.store_id]["memory"], item.store_id, query_text, top_k, item_id=item.item_id
            ),
        ),
        (
            "amem",
            lambda: amem_adapter.retrieve(
                amem_stores[item.store_id]["system"],
                amem_stores[item.store_id]["id_map"],
                query_text,
                top_k,
                item_id=item.item_id,
                store_id=item.store_id,
            ),
        ),
    )

    for system_name, retrieve_fn in systems:
        retrieved = retrieve_fn()
        retrieved_ids = [r["id"] for r in retrieved]

        base_block = "\n".join(r["text"] for r in retrieved) if retrieved else None
        base_text, base_correct = _answer(client, item.current_fact, item.question, base_block, item.new_target_forms)
        predictions.append(
            {
                "item_id": item.item_id,
                "edit_id": item.edit_id,
                "condition": item.condition,
                "relation": item.relation,
                "method": "base",
                "memory_system": system_name,
                "retrieved_ids": retrieved_ids,
                "memory_block": base_block,
                "raw_output": base_text,
                "correct": base_correct,
            }
        )

        store_trie = tries[system_name][item.store_id]
        store_parsed = parsed[system_name][item.store_id]
        retrieved_records = [store_parsed.get(r["id"]) for r in retrieved]
        retrieved_records = [r for r in retrieved_records if r is not None and r.parse_ok]
        memtrim_block = memtrim.build_memtrim_block(query_record, retrieved_records, store_trie, flags, embed_fn=encoder.encode)
        memtrim_block = memtrim_block or None
        memtrim_text, memtrim_correct = _answer(client, item.current_fact, item.question, memtrim_block, item.new_target_forms)
        predictions.append(
            {
                "item_id": item.item_id,
                "edit_id": item.edit_id,
                "condition": item.condition,
                "relation": item.relation,
                "method": "memtrim",
                "memory_system": system_name,
                "retrieved_ids": retrieved_ids,
                "memory_block": memtrim_block,
                "raw_output": memtrim_text,
                "correct": memtrim_correct,
            }
        )

    return predictions


def run_answering(client, items, mem0_stores, amem_stores, tries, parsed, flags, top_k, workers):
    no_memory_cache = {}
    no_memory_lock = threading.Lock()
    all_predictions = []

    def work(item):
        return process_item(client, item, mem0_stores, amem_stores, tries, parsed, no_memory_cache, no_memory_lock, flags, top_k)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, item) for item in items]
        for fut in as_completed(futures):
            all_predictions.extend(fut.result())

    return all_predictions


def save_predictions(predictions, path):
    with open(path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False))
            f.write("\n")


def save_dataset(items, stores, items_path, stores_path):
    with open(items_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "item_id": it.item_id,
                    "edit_id": it.edit_id,
                    "condition": it.condition,
                    "relation": it.relation,
                    "subject_id": it.subject_id,
                    "current_fact": it.current_fact,
                    "question": it.question,
                    "new_target_forms": sorted(it.new_target_forms),
                    "memory_text": it.memory_text,
                    "unchanged_facts": it.unchanged_facts,
                    "store_id": it.store_id,
                }
                for it in items
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(stores_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                store_id: [
                    {
                        "interaction_id": ia.interaction_id,
                        "text": ia.text,
                        "relation": ia.relation,
                        "subject_id": ia.subject_id,
                        "item_id": ia.item_id,
                        "is_primary": ia.is_primary,
                    }
                    for ia in interactions
                ]
                for store_id, interactions in stores.items()
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
