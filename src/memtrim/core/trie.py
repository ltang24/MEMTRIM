import numpy as np

from . import alias
from .parser import normalize_value

DETERMINISTIC_VALUE_TYPES = ("boolean", "numeric", "temporal", "categorical")


def _cosine(a, b):
    return float(np.dot(a, b))


class KeyRegistry:
    def __init__(self, embed_fn=None, equiv_fn=None, similarity_threshold=0.45, max_candidates=8):
        self.embed_fn = embed_fn
        self.equiv_fn = equiv_fn
        self.similarity_threshold = similarity_threshold
        self.max_candidates = max_candidates
        self.keys = []
        self._exact_index = {}

    def _update_multi_valued(self, key_id, multi_valued_hint):
        for k in self.keys:
            if k["id"] == key_id and multi_valued_hint:
                k["multi_valued"] = True

    def is_multi_valued(self, key_id):
        for k in self.keys:
            if k["id"] == key_id:
                return k["multi_valued"]
        return False

    def canonicalize(self, entity, attribute, value_type, multi_valued_hint, create):
        norm = (alias.normalize(entity), alias.normalize(attribute))
        exact = self._exact_index.get(norm)
        if exact is not None:
            self._update_multi_valued(exact, multi_valued_hint)
            return exact

        same_type = [k for k in self.keys if k["value_type"] == value_type]
        if self.embed_fn is not None and same_type:
            query_vec = self.embed_fn(f"{entity} {attribute}")
            scored = sorted(same_type, key=lambda k: -_cosine(query_vec, k["embedding"]))
            shortlist = [k for k in scored[: self.max_candidates] if _cosine(query_vec, k["embedding"]) >= self.similarity_threshold]
        else:
            shortlist = same_type[: self.max_candidates]

        chosen = None
        if shortlist and self.equiv_fn is not None:
            candidates = [{"entity": k["entity"], "attribute": k["attribute"], "value_type": k["value_type"]} for k in shortlist]
            idx = self.equiv_fn(entity, attribute, value_type, candidates)
            if idx is not None and 0 <= idx < len(shortlist):
                chosen = shortlist[idx]

        if chosen is not None:
            self._exact_index[norm] = chosen["id"]
            self._update_multi_valued(chosen["id"], multi_valued_hint)
            return chosen["id"]

        if not create:
            return None

        new_id = f"k{len(self.keys)}"
        embedding = self.embed_fn(f"{entity} {attribute}") if self.embed_fn is not None else None
        self.keys.append(
            {
                "id": new_id,
                "entity": entity,
                "attribute": attribute,
                "value_type": value_type,
                "multi_valued": multi_valued_hint,
                "embedding": embedding,
            }
        )
        self._exact_index[norm] = new_id
        return new_id


class ValueRegistry:
    def __init__(self, equiv_fn=None):
        self.equiv_fn = equiv_fn
        self.values = []
        self._exact_index = {}

    def canonicalize(self, key_label, value_type, raw_value, create):
        normalized = normalize_value(value_type, raw_value)
        exact = self._exact_index.get(normalized)
        if exact is not None:
            return exact

        if value_type in DETERMINISTIC_VALUE_TYPES:
            if not create:
                return None
            new_id = f"v{len(self.values)}"
            self.values.append({"id": new_id, "raw": raw_value, "normalized": normalized})
            self._exact_index[normalized] = new_id
            return new_id

        chosen = None
        if self.values and self.equiv_fn is not None:
            idx = self.equiv_fn(key_label, raw_value, [v["raw"] for v in self.values])
            if idx is not None and 0 <= idx < len(self.values):
                chosen = self.values[idx]
        if chosen is not None:
            self._exact_index[normalized] = chosen["id"]
            return chosen["id"]

        if not create:
            return None
        new_id = f"v{len(self.values)}"
        self.values.append({"id": new_id, "raw": raw_value, "normalized": normalized})
        self._exact_index[normalized] = new_id
        return new_id


class TaskRegistry:
    def __init__(self, equiv_fn=None):
        self.equiv_fn = equiv_fn
        self.tasks = []
        self._exact_index = {}

    def canonicalize(self, operation, target, create):
        norm = (alias.normalize(operation), alias.normalize(target))
        exact = self._exact_index.get(norm)
        if exact is not None:
            return exact

        chosen = None
        if self.tasks and self.equiv_fn is not None:
            candidates = [{"operation": t["operation"], "target": t["target"]} for t in self.tasks]
            idx = self.equiv_fn(operation, target, candidates)
            if idx is not None and 0 <= idx < len(self.tasks):
                chosen = self.tasks[idx]
        if chosen is not None:
            self._exact_index[norm] = chosen["id"]
            return chosen["id"]

        if not create:
            return None
        new_id = f"t{len(self.tasks)}"
        self.tasks.append({"id": new_id, "operation": operation, "target": target})
        self._exact_index[norm] = new_id
        return new_id


class Trie:
    def __init__(self, embed_fn=None, equiv_key_fn=None, equiv_value_fn=None, equiv_task_fn=None):
        self.keys = KeyRegistry(embed_fn=embed_fn, equiv_fn=equiv_key_fn)
        self.values = {}
        self.equiv_value_fn = equiv_value_fn
        self.tasks = TaskRegistry(equiv_fn=equiv_task_fn)
        self.leaves = {}

    def _value_registry(self, key_id):
        if key_id not in self.values:
            self.values[key_id] = ValueRegistry(equiv_fn=self.equiv_value_fn)
        return self.values[key_id]

    def is_multi_valued(self, key_id):
        return self.keys.is_multi_valued(key_id)

    def insert(self, unit, memory_id, in_support):
        key_id = self.keys.canonicalize(unit.entity, unit.attribute, unit.value_type, unit.multi_valued, create=True)
        key_label = f"{unit.entity}/{unit.attribute}"
        value_id = self._value_registry(key_id).canonicalize(key_label, unit.value_type, unit.value, create=True)
        unit.key_id = key_id
        unit.value_id = value_id
        leaf = self.leaves.setdefault((key_id, value_id), {"memory_ids": set(), "support_memory_ids": set()})
        leaf["memory_ids"].add(memory_id)
        if in_support:
            leaf["support_memory_ids"].add(memory_id)
        return key_id, value_id

    def match_query_unit(self, unit):
        key_id = self.keys.canonicalize(unit.entity, unit.attribute, unit.value_type, unit.multi_valued, create=False)
        value_id = None
        if key_id is not None:
            key_label = f"{unit.entity}/{unit.attribute}"
            value_id = self._value_registry(key_id).canonicalize(key_label, unit.value_type, unit.value, create=False)
        unit.key_id = key_id
        unit.value_id = value_id
        return key_id, value_id

    def match_task(self, task):
        return self.tasks.canonicalize(task.operation, task.target, create=False)

    def insert_task(self, task):
        task_id = self.tasks.canonicalize(task.operation, task.target, create=True)
        task.task_id = task_id
        return task_id

    def memory_ids_for(self, key_id, value_id):
        leaf = self.leaves.get((key_id, value_id))
        return set(leaf["memory_ids"]) if leaf else set()
