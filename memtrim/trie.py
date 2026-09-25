"""A minimal key-to-value-to-memory-ID evidence trie."""

from __future__ import annotations

from .schema import EvidenceUnit


class EvidenceTrie:
    """Index independent evidence units without performing retrieval."""

    def __init__(self) -> None:
        self.root: dict[str, dict[str, set[str]]] = {}

    def insert(self, evidence_unit: EvidenceUnit, memory_id: str) -> None:
        """Index one evidence unit at write time under its key and value."""
        values = self.root.setdefault(evidence_unit.key, {})
        values.setdefault(evidence_unit.value, set()).add(memory_id)

    def exact_lookup(self, key: str, value: str) -> set[str]:
        """Return memory IDs containing exactly this key-value pair."""
        return set(self.root.get(key, {}).get(value, set()))

    def values_for_key(self, key: str) -> set[str]:
        """Return the distinct indexed values for a canonical key."""
        return set(self.root.get(key, {}))

    def memories_for_key(self, key: str) -> set[str]:
        """Return memory IDs containing any value for a canonical key."""
        memory_ids: set[str] = set()
        for ids_for_value in self.root.get(key, {}).values():
            memory_ids.update(ids_for_value)
        return memory_ids
