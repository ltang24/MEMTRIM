

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import EvidenceUnit

KEY_SEP = "|"


@dataclass
class _Node:
    children: dict[str, "_Node"] = field(default_factory=dict)
    # Populated only on value leaves: memory_id -> in_support
    memories: dict[str, bool] = field(default_factory=dict)


class EvidenceTrie:
    """Index independent evidence units; never used as a retriever."""

    def __init__(self) -> None:
        self.root = _Node()
        self._n_units = 0

    @staticmethod
    def _path(key: str) -> list[str]:
        return [seg for seg in key.split(KEY_SEP)] or [key]

    def _key_node(self, key: str, create: bool) -> _Node | None:
        node = self.root
        for seg in self._path(key):
            nxt = node.children.get(seg)
            if nxt is None:
                if not create:
                    return None
                nxt = node.children[seg] = _Node()
            node = nxt
        return node

    def insert(self, evidence_unit: EvidenceUnit, memory_id: str, in_support: bool = False) -> None:
        key_node = self._key_node(evidence_unit.key, create=True)
        leaf = key_node.children.setdefault("=" + evidence_unit.value, _Node())
        leaf.memories[memory_id] = leaf.memories.get(memory_id, False) or in_support
        self._n_units += 1

    def remove_memory(self, memory_id: str) -> None:
        def walk(node: _Node) -> None:
            node.memories.pop(memory_id, None)
            for child in node.children.values():
                walk(child)

        walk(self.root)

    # ------------------------------------------------------------ lookups
    def exact_lookup(self, key: str, value: str) -> set[str]:
        """Memory IDs containing exactly this key-value pair."""
        key_node = self._key_node(key, create=False)
        if key_node is None:
            return set()
        leaf = key_node.children.get("=" + value)
        return set(leaf.memories) if leaf else set()

    def values_for_key(self, key: str) -> set[str]:
        key_node = self._key_node(key, create=False)
        if key_node is None:
            return set()
        return {k[1:] for k in key_node.children if k.startswith("=")}

    def memories_for_key(self, key: str) -> set[str]:
        key_node = self._key_node(key, create=False)
        if key_node is None:
            return set()
        out: set[str] = set()
        for k, leaf in key_node.children.items():
            if k.startswith("="):
                out.update(leaf.memories)
        return out

    def is_support(self, key: str, value: str, memory_id: str) -> bool:
        key_node = self._key_node(key, create=False)
        leaf = key_node.children.get("=" + value) if key_node else None
        return bool(leaf and leaf.memories.get(memory_id, False))

    def __len__(self) -> int:
        return self._n_units
