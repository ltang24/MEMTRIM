import importlib
import json
import os

from ..core import config
from .base import Interaction, Item

DATASETS = {
    "rippleedits": "memtrim.datasets.rippleedits",
    "counterlogic": "memtrim.datasets.counterlogic",
    "cladder": "memtrim.datasets.cladder",
}


def build(name, **kwargs):
    return importlib.import_module(DATASETS[name]).build(**kwargs)


def data_dir(name, root=None):
    return os.path.join(root or config.DATA_DIR, name)


def settings(name, seed, n, max_store_size, params):
    return {"dataset": name, "seed": seed, "n_per_condition": n, "max_store_size": max_store_size, "params": dict(sorted(params.items()))}


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def save(name, meta, items, stores, root=None):
    d = data_dir(name, root)
    os.makedirs(d, exist_ok=True)
    _write_json(os.path.join(d, "items.json"), {"settings": meta, "items": [it.to_dict() for it in items]})
    _write_json(
        os.path.join(d, "stores.json"),
        {"settings": meta, "stores": {sid: [ia.to_dict() for ia in ias] for sid, ias in stores.items()}},
    )
    return d


def load(name, meta, root=None):
    d = data_dir(name, root)
    paths = [os.path.join(d, "items.json"), os.path.join(d, "stores.json")]
    if not all(os.path.exists(p) for p in paths):
        return None
    with open(paths[0], "r", encoding="utf-8") as f:
        items_doc = json.load(f)
    with open(paths[1], "r", encoding="utf-8") as f:
        stores_doc = json.load(f)
    if items_doc.get("settings") != meta or stores_doc.get("settings") != meta:
        return None
    items = [Item(**d) for d in items_doc["items"]]
    stores = {sid: [Interaction(**d) for d in ias] for sid, ias in stores_doc["stores"].items()}
    return items, stores


def load_or_build(name, seed, n, max_store_size, params, root=None):
    meta = settings(name, seed, n, max_store_size, params)
    loaded = load(name, meta, root)
    if loaded is not None:
        return loaded
    return build(name, seed=seed, n=n, max_store_size=max_store_size, **params)
