import json
import os
import random

from ..core import alias, config
from .base import TA, TD_DUP, Item, build_stores


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
        "new_target_aliases": list(gold_answers[0].get("aliases") or []),
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


def sample_candidates(n, seed):
    candidates = load_candidates()
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[: min(n, len(candidates))]


def memory_of(cand, condition):
    if condition == TA:
        return {"text": f"{cand['current_fact']} {cand['question']}", "answer": cand["new_target_display"]}
    return {"text": f"{cand['old_fact_text']} {cand['question']}", "answer": cand["old_target"]}


def make_items(sampled):
    items = []
    for i, cand in enumerate(sampled):
        pair_id = f"e{i}"
        for condition in (TA, TD_DUP):
            items.append(
                Item(
                    id=f"{pair_id}_{condition}",
                    pair_id=pair_id,
                    dataset="rippleedits",
                    condition=condition,
                    group=cand["relation"],
                    query=f"{cand['current_fact']} {cand['question']}",
                    gold=cand["new_target_display"],
                    aliases=list(cand["new_target_aliases"]),
                    answer_type="entity",
                    memory=memory_of(cand, condition),
                    extra_memories=[{"text": f, "answer": None} for f in cand["unchanged_facts"]],
                )
            )
    return items


def build(n=config.DEFAULT_N_PER_CONDITION, seed=config.DEFAULT_SEED, max_store_size=config.DEFAULT_MAX_STORE_SIZE, **_):
    items = make_items(sample_candidates(n, seed))
    stores = build_stores(items, max_store_size)
    return items, stores
