import json
from collections import Counter

from ..core import config
from .base import TA, TD_DUP, Item, build_stores, sample_per_condition

MEMORY_QUESTION = "Is the argument logically valid?"
QUERY_QUESTION = "Does the conclusion agree with real-world knowledge?"


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for key in sorted(data, key=lambda k: (len(k), k)):
        for i, record in enumerate(data[key]):
            records.append(dict(record, id=f"cl_{key}_{i}"))
    return records


def yes_no(flag):
    return "yes" if flag else "no"


def argument_text(record):
    premises = " ".join(p.strip().rstrip(".") + "." for p in record["premise_list"])
    return f"Premises: {premises} Conclusion: {record['conclusion'].strip().rstrip('.')}."


def embedding_text(record):
    return " ".join(record["premise_list"]) + " " + record["conclusion"]


def condition_of(record):
    return TA if bool(record["validity"]) == bool(record["believability"]) else TD_DUP


def make_item(record, group):
    argument = argument_text(record)
    return Item(
        id=record["id"],
        pair_id=record["id"],
        dataset="counterlogic",
        condition=condition_of(record),
        group=group,
        query=f"{argument} {QUERY_QUESTION}",
        gold=yes_no(record["believability"]),
        answer_type="yes_no",
        memory={"text": f"{argument} {MEMORY_QUESTION}", "answer": yes_no(record["validity"])},
    )


def cluster_groups(texts, embed_fn, distance_threshold=0.2, min_cluster_size=5):
    from sklearn.cluster import AgglomerativeClustering

    if len(texts) < 2:
        return [None] * len(texts)
    vectors = embed_fn(list(texts))
    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=distance_threshold
    ).fit_predict(vectors)
    sizes = Counter(labels)
    return [f"c{label}" if sizes[label] >= min_cluster_size else None for label in labels]


def build(path=None, n=500, seed=config.DEFAULT_SEED, max_store_size=config.DEFAULT_MAX_STORE_SIZE,
          distance_threshold=0.2, min_cluster_size=5, embed_fn=None, **_):
    records = load_records(path or config.COUNTERLOGIC_PATH)
    if embed_fn is None:
        from ..core.encoder import encode as embed_fn
    groups = cluster_groups([embedding_text(r) for r in records], embed_fn, distance_threshold, min_cluster_size)
    items = [make_item(r, g) for r, g in zip(records, groups) if g is not None]
    items = sample_per_condition(items, n, seed)
    stores = build_stores(items, max_store_size)
    return items, stores
