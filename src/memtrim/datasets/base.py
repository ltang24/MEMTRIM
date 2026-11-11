import random
from dataclasses import asdict, dataclass, field
from typing import List, Optional

TA = "TA"
TD_DUP = "TD_DUP"
CONDITIONS = (TA, TD_DUP)


@dataclass
class Item:
    id: str
    pair_id: str
    dataset: str
    condition: str
    group: str
    query: str
    gold: str
    aliases: List[str] = field(default_factory=list)
    answer_type: str = "entity"
    memory: dict = field(default_factory=dict)
    store_id: str = ""
    extra_memories: List[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class Interaction:
    interaction_id: str
    item_id: str
    group: str
    text: str
    answer: Optional[str] = None
    is_primary: bool = True

    def to_dict(self):
        return asdict(self)


def compose(text, answer):
    return text if not answer else f"{text} {answer}."


def sample_per_condition(items, n, seed):
    rng = random.Random(seed)
    out = []
    for condition in CONDITIONS:
        pool = sorted((it for it in items if it.condition == condition), key=lambda it: it.id)
        rng.shuffle(pool)
        out.extend(pool[:n])
    return out


def item_interactions(item):
    primary = Interaction(
        interaction_id=f"{item.id}_primary",
        item_id=item.id,
        group=item.group,
        text=item.memory["text"],
        answer=item.memory.get("answer"),
        is_primary=True,
    )
    extras = [
        Interaction(
            interaction_id=f"{item.id}_extra{j}",
            item_id=item.id,
            group=item.group,
            text=m["text"],
            answer=m.get("answer"),
            is_primary=False,
        )
        for j, m in enumerate(item.extra_memories)
    ]
    return [primary] + extras


def build_stores(items, max_store_size):
    groups = {}
    for item in items:
        groups.setdefault((item.group, item.condition), []).append(item)
    stores = {}
    for (group, condition), group_items in groups.items():
        idx = 0
        store_id = f"{group}__{condition}__{idx}"
        current = []
        for item in group_items:
            new = item_interactions(item)
            if current and len(current) + len(new) > max_store_size:
                stores[store_id] = current
                idx += 1
                store_id = f"{group}__{condition}__{idx}"
                current = []
            current.extend(new)
            item.store_id = store_id
        if current:
            stores[store_id] = current
    return stores
