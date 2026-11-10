import json
import os
import random

from . import alias, config
from .schema import Item, StoredInteraction


def _common_prefix_word_count(a_words, b_words):
    n = min(len(a_words), len(b_words))
    i = 0
    while i < n and a_words[i] == b_words[i]:
        i += 1
    return i


def _extract_target(full_text, prefix_word_count):
    words = full_text.split()
    tail = " ".join(words[prefix_word_count:]).strip()
    if tail.endswith("."):
        tail = tail[:-1]
    return tail.strip()


def _load_raw_edits():
    entries = []
    for fname in ("random.json", "popular.json"):
        path = os.path.join(config.RIPPLEEDITS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            entries.extend(json.load(f))
    return entries


def _unchanged_facts(raw_entry, limit):
    facts = []
    for axis in ("Relation_Specificity", "Forgetfulness"):
        for group in raw_entry.get(axis) or []:
            for q in group.get("test_queries") or []:
                answers = q.get("answers") or []
                prompt = q.get("prompt")
                if not answers or not prompt:
                    continue
                value = answers[0].get("value")
                if not value:
                    continue
                sentence = f"{prompt} {value}."
                if sentence not in facts:
                    facts.append(sentence)
                if len(facts) >= limit:
                    return facts
    return facts


def _build_candidate(raw_entry):
    edit = raw_entry["edit"]
    original = edit.get("original_fact")
    if original is None:
        return None
    sa = raw_entry.get("Subject_Aliasing") or []
    if not sa or not sa[0].get("test_queries"):
        return None
    sa_query = sa[0]["test_queries"][0]
    gold_answers = sa_query.get("answers") or []
    question = sa_query.get("prompt")
    if not gold_answers or not question:
        return None

    new_text = edit["prompt"]
    old_text = original["prompt"]
    prefix_n = _common_prefix_word_count(new_text.split(), old_text.split())
    new_target = _extract_target(new_text, prefix_n)
    old_target = _extract_target(old_text, prefix_n)

    new_target_forms = alias.gold_forms(gold_answers[0]["value"], gold_answers[0].get("aliases"))
    if not new_target_forms or alias.normalize(new_target) not in new_target_forms:
        return None

    old_target_forms = alias.gold_forms(old_target)
    if not old_target or alias.forms_equal(old_target_forms, new_target_forms):
        return None

    return {
        "relation": edit["relation"],
        "subject_id": edit["subject_id"],
        "current_fact": new_text,
        "question": question,
        "new_target_forms": new_target_forms,
        "new_target_display": gold_answers[0]["value"],
        "old_target": old_target,
        "old_fact_text": old_text,
        "unchanged_facts": _unchanged_facts(raw_entry, config.MAX_UNCHANGED_FACTS),
    }


def load_candidates():
    candidates = []
    for raw_entry in _load_raw_edits():
        candidate = _build_candidate(raw_entry)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def sample_items(n, seed=config.DEFAULT_SEED):
    candidates = load_candidates()
    rng = random.Random(seed)
    rng.shuffle(candidates)
    sampled = candidates[: min(n, len(candidates))]

    items = []
    for i, cand in enumerate(sampled):
        edit_id = f"e{i}"
        tddup_memory_text = f"{cand['old_fact_text']} {cand['question']} {cand['old_target']}."
        for condition, memory_text in (("TA", _ta_memory_sentence(cand)), ("TD_DUP", tddup_memory_text)):
            items.append(
                Item(
                    item_id=f"{edit_id}_{condition}",
                    edit_id=edit_id,
                    condition=condition,
                    relation=cand["relation"],
                    subject_id=cand["subject_id"],
                    current_fact=cand["current_fact"],
                    question=cand["question"],
                    new_target_forms=cand["new_target_forms"],
                    memory_text=memory_text,
                    unchanged_facts=list(cand["unchanged_facts"]),
                )
            )
    return items


def _ta_memory_sentence(cand):
    return f"{cand['current_fact']} {cand['question']} {_new_target_display(cand)}."


def _new_target_display(cand):
    return cand["new_target_display"]


def build_stores(items, max_store_size=config.DEFAULT_MAX_STORE_SIZE):
    groups = {}
    for item in items:
        groups.setdefault((item.relation, item.condition), []).append(item)

    stores = {}
    for (relation, condition), group_items in groups.items():
        store_idx = 0
        current_store_id = f"{relation}__{condition}__{store_idx}"
        current_interactions = []
        for item in group_items:
            new_interactions = [
                StoredInteraction(
                    interaction_id=f"{item.item_id}_primary",
                    text=item.memory_text,
                    relation=relation,
                    subject_id=item.subject_id,
                    item_id=item.item_id,
                    is_primary=True,
                )
            ]
            for j, fact in enumerate(item.unchanged_facts):
                new_interactions.append(
                    StoredInteraction(
                        interaction_id=f"{item.item_id}_fact{j}",
                        text=fact,
                        relation=relation,
                        subject_id=item.subject_id,
                        item_id=item.item_id,
                        is_primary=False,
                    )
                )

            if current_interactions and len(current_interactions) + len(new_interactions) > max_store_size:
                stores[current_store_id] = current_interactions
                store_idx += 1
                current_store_id = f"{relation}__{condition}__{store_idx}"
                current_interactions = []

            current_interactions.extend(new_interactions)
            item.store_id = current_store_id

        if current_interactions:
            stores[current_store_id] = current_interactions

    return stores
