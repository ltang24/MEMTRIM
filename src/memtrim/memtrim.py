from dataclasses import dataclass

import numpy as np

from . import alias

SHARED = "shared"
CONFLICTING = "conflicting"
MEMORY_ONLY = "memory_only"


@dataclass
class AblationFlags:
    keep_shared: bool = False
    keep_conflicts: bool = False
    keep_old_answer: bool = False
    ignore_task_change: bool = False
    embedding_match: bool = False


def build_query_context(query_record, trie):
    query_key_values = {}
    query_keys_present = set()
    for unit in query_record.evidence:
        key_id, value_id = trie.match_query_unit(unit)
        if key_id is None:
            continue
        query_keys_present.add(key_id)
        if value_id is not None:
            query_key_values.setdefault(key_id, set()).add(value_id)
    query_task_id = trie.match_task(query_record.task) if query_record.task is not None else None
    return query_key_values, query_keys_present, query_task_id


def classify_unit(unit, query_key_values, query_keys_present, trie):
    if unit.key_id is None or unit.key_id not in query_keys_present:
        return MEMORY_ONLY
    if unit.value_id in query_key_values.get(unit.key_id, set()):
        return SHARED
    if trie.is_multi_valued(unit.key_id):
        return MEMORY_ONLY
    return CONFLICTING


def _cosine(a, b):
    return float(np.dot(a, b))


def classify_unit_embedding(unit, query_units, embed_fn, threshold=0.85):
    unit_vec = embed_fn(f"{unit.entity} {unit.attribute}")
    best, best_sim = None, -1.0
    for q in query_units:
        sim = _cosine(unit_vec, q["embedding"])
        if sim > best_sim:
            best_sim, best = sim, q
    if best is None or best_sim < threshold:
        return MEMORY_ONLY
    if unit.value_type == best["unit"].value_type and alias.normalize(unit.value) == alias.normalize(best["unit"].value):
        return SHARED
    if unit.multi_valued:
        return MEMORY_ONLY
    return CONFLICTING


def classify_memory(memory_record, query_record, query_key_values, query_keys_present, trie, flags, embed_fn=None):
    if flags.embedding_match and embed_fn is not None:
        query_units = [{"unit": u, "embedding": embed_fn(f"{u.entity} {u.attribute}")} for u in query_record.evidence]
        return [classify_unit_embedding(u, query_units, embed_fn) for u in memory_record.evidence]
    return [classify_unit(u, query_key_values, query_keys_present, trie) for u in memory_record.evidence]


def tasks_match(memory_task, query_task_id):
    return memory_task is not None and memory_task.task_id is not None and memory_task.task_id == query_task_id


def transform_memory(memory_record, classifications, query_task_id, flags):
    kept_units = []
    for unit, cls in zip(memory_record.evidence, classifications):
        if cls == MEMORY_ONLY:
            kept_units.append(unit)
        elif cls == SHARED and flags.keep_shared:
            kept_units.append(unit)
        elif cls == CONFLICTING and flags.keep_conflicts:
            kept_units.append(unit)

    support_conflict = any(
        cls == CONFLICTING for cls, supported in zip(classifications, memory_record.support_mask) if supported
    )
    suppress = support_conflict and not flags.keep_old_answer

    outcome_contribution = None
    if memory_record.outcome_text is not None and not suppress:
        same_task = tasks_match(memory_record.task, query_task_id)
        if same_task or flags.ignore_task_change:
            outcome_contribution = ("answer", memory_record.outcome_text)
        elif memory_record.outcome_evidence is not None:
            outcome_contribution = ("labeled_fact", memory_record.outcome_evidence, memory_record.task)

    return kept_units, outcome_contribution


def merge_contexts(per_memory_results, trie):
    seen_pairs = set()
    by_key = {}
    answers = []

    for kept_units, outcome in per_memory_results:
        for unit in kept_units:
            pair = (unit.key_id, unit.value_id)
            if pair in seen_pairs or unit.key_id is None:
                continue
            seen_pairs.add(pair)
            by_key.setdefault(unit.key_id, []).append((unit.value_id, unit))
        if outcome is not None:
            if outcome[0] == "answer":
                answers.append(outcome[1])
            elif outcome[0] == "labeled_fact":
                fact_unit = outcome[1]
                pair = (fact_unit.key_id, fact_unit.value_id)
                if fact_unit.key_id is not None and pair not in seen_pairs:
                    seen_pairs.add(pair)
                    by_key.setdefault(fact_unit.key_id, []).append((fact_unit.value_id, outcome))

    final_items = []
    for key_id, entries in by_key.items():
        distinct_values = {value_id for value_id, _ in entries}
        if len(distinct_values) > 1 and not trie.is_multi_valued(key_id):
            continue
        for _, payload in entries:
            final_items.append(payload)

    kept_answer = None
    if answers:
        normalized = {alias.normalize(a) for a in answers}
        if len(normalized) == 1:
            kept_answer = answers[0]

    return final_items, kept_answer


def render_context(final_items, kept_answer):
    lines = []
    for item in final_items:
        if isinstance(item, tuple) and item[0] == "labeled_fact":
            fact_unit, task = item[1], item[2]
            lines.append(
                f"From an earlier interaction about {task.operation}/{task.target}: {fact_unit.entity} {fact_unit.attribute} = {fact_unit.value}"
            )
        else:
            lines.append(f"{item.entity} {item.attribute} = {item.value}")
    if kept_answer is not None:
        lines.append(f"Previously stored answer to this same question: {kept_answer}")
    return "\n".join(lines)


def build_memtrim_block(query_record, retrieved_records, trie, flags=None, embed_fn=None):
    flags = flags or AblationFlags()
    query_key_values, query_keys_present, query_task_id = build_query_context(query_record, trie)
    per_memory_results = []
    for memory_record in retrieved_records:
        classifications = classify_memory(
            memory_record, query_record, query_key_values, query_keys_present, trie, flags, embed_fn=embed_fn
        )
        kept_units, outcome = transform_memory(memory_record, classifications, query_task_id, flags)
        per_memory_results.append((kept_units, outcome))
    final_items, kept_answer = merge_contexts(per_memory_results, trie)
    return render_context(final_items, kept_answer)
