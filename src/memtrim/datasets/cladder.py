import difflib
import json
import re
import zipfile

from ..core import config
from .base import TA, TD_DUP, Item, build_stores, sample_per_condition

QUESTIONS_FILE = "cladder-v1-q-balanced.json"
MODELS_FILE = "cladder-v1-meta-models.json"
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def load_records(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        questions = json.loads(z.read(QUESTIONS_FILE))
        models = {m["model_id"]: m for m in json.loads(z.read(MODELS_FILE))}
    for q in questions:
        q["background"] = models[q["meta"]["model_id"]]["background"]
    return questions


def group_key(record):
    meta = record["meta"]
    return f"{meta['story_id']}-{meta['graph_id']}-{meta['query_type']}-r{meta['rung']}"


def full_text(record):
    return f"{record['background']} {record['given_info']} {record['question']}"


def mask_numbers(text):
    return _NUMBER_RE.sub("#", text)


def masked_similarity(a, b):
    return difflib.SequenceMatcher(None, mask_numbers(a), mask_numbers(b), autojunk=False).ratio()


def _flatten(value, path=()):
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _flatten(v, path + (i,))
    elif isinstance(value, dict):
        for k in sorted(value):
            yield from _flatten(value[k], path + (k,))
    elif isinstance(value, (int, float)):
        yield path, round(float(value), 2)


def leaves(given_info):
    return {(key,) + path: v for key, value in given_info.items() for path, v in _flatten(value)}


def key_jaccard(a, b):
    ka, kb = set(a), set(b)
    union = ka | kb
    return len(ka & kb) / len(union) if union else 0.0


def value_changes(source_info, candidate_info):
    shared = set(source_info) & set(candidate_info)
    src = {p: v for p, v in leaves(source_info).items() if p[0] in shared}
    cand = {p: v for p, v in leaves(candidate_info).items() if p[0] in shared}
    unchanged, changed = [], []
    for path in sorted(set(src) & set(cand), key=str):
        (unchanged if src[path] == cand[path] else changed).append(cand[path])
    return unchanged, changed


def reasoning_text(record):
    return " ".join(str(v) for v in (record.get("reasoning") or {}).values())


def numeric_info(record):
    info = record["meta"].get("given_info")
    return info if isinstance(info, dict) else {}


def is_pair(source, candidate, similarity_threshold=0.85, jaccard_threshold=0.8):
    if source["question_id"] == candidate["question_id"] or source["question"] != candidate["question"]:
        return False
    s_info, c_info = numeric_info(source), numeric_info(candidate)
    if key_jaccard(s_info, c_info) < jaccard_threshold:
        return False
    unchanged, changed = value_changes(s_info, c_info)
    if not unchanged or not changed:
        return False
    reasoning = reasoning_text(candidate)
    if not any(f"{v:.2f}" in reasoning for v in changed):
        return False
    return masked_similarity(source["given_info"], candidate["given_info"]) >= similarity_threshold


def find_pairs(records, similarity_threshold=0.85, jaccard_threshold=0.8):
    groups = {}
    for r in records:
        groups.setdefault(group_key(r), []).append(r)
    pairs = []
    for key in sorted(groups):
        members = groups[key]
        for s in members:
            for c in members:
                if is_pair(s, c, similarity_threshold, jaccard_threshold):
                    pairs.append((key, s, c))
    return pairs


def make_item(group, source, candidate):
    pair_id = f"{source['question_id']}-{candidate['question_id']}"
    return Item(
        id=f"cladder_{pair_id}",
        pair_id=pair_id,
        dataset="cladder",
        condition=TA if source["answer"] == candidate["answer"] else TD_DUP,
        group=group,
        query=full_text(candidate),
        gold=candidate["answer"],
        answer_type="yes_no",
        memory={"text": full_text(source), "answer": source["answer"]},
    )


def build(path=None, n=500, seed=config.DEFAULT_SEED, max_store_size=config.DEFAULT_MAX_STORE_SIZE,
          similarity_threshold=0.85, jaccard_threshold=0.8, **_):
    records = load_records(path or config.CLADDER_ZIP_PATH)
    pairs = find_pairs(records, similarity_threshold, jaccard_threshold)
    items = sample_per_condition([make_item(*p) for p in pairs], n, seed)
    stores = build_stores(items, max_store_size)
    return items, stores
