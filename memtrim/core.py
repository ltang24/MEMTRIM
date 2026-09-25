"""Write-time indexing and read-time transformation of retrieved memories."""

from __future__ import annotations

from typing import TypedDict

from .schema import EvidenceUnit, MemoryRecord, MemoryTransformResult, ParsedQuery
from .trie import EvidenceTrie


class MemoryContext(TypedDict):
    """Merged evidence and stored outcomes for the caller to render."""

    evidence: list[EvidenceUnit]
    outcomes: list[str]


class MemTrim:
    """Transform memories selected by an external memory retriever."""

    def __init__(self) -> None:
        self.memories: dict[str, MemoryRecord] = {}
        self.trie = EvidenceTrie()

    def add_memory(self, memory: MemoryRecord) -> None:
        """Cache a new memory and independently index each unit at write time.

        Duplicate IDs raise ValueError to prevent stale index entries.
        """
        if memory.memory_id in self.memories:
            raise ValueError(f"Memory ID already exists: {memory.memory_id}")
        self.memories[memory.memory_id] = memory
        for unit in memory.evidence:
            self.trie.insert(unit, memory.memory_id)

    def transform_memory(
        self, query: ParsedQuery, memory: MemoryRecord
    ) -> MemoryTransformResult:
        """Partition evidence into shared, conflicting, and memory-only units.

        Exact matches are shared, even if a query lists other values for that
        key. Only memory-only units survive. A conflict with the canonical key
        of support evidence suppresses both outcome forms before task matching.
        """
        query_values: dict[str, set[str]] = {}
        for unit in query.evidence:
            query_values.setdefault(unit.key, set()).add(unit.value)

        shared: list[EvidenceUnit] = []
        conflicting: list[EvidenceUnit] = []
        memory_only: list[EvidenceUnit] = []
        for unit in memory.evidence:
            if unit.key not in query_values:
                memory_only.append(unit)
            elif unit.value in query_values[unit.key]:
                shared.append(unit)
            else:
                conflicting.append(unit)

        support_keys = {unit.key for unit in memory.support_evidence}
        outcome_suppressed = any(unit.key in support_keys for unit in conflicting)
        retained_outcome: str | None = None
        outcome_evidence: EvidenceUnit | None = None
        if not outcome_suppressed:
            if memory.task_signature == query.task_signature:
                retained_outcome = memory.outcome
            else:
                outcome_evidence = memory.outcome_evidence

        return MemoryTransformResult(
            memory_id=memory.memory_id,
            shared=shared,
            conflicting=conflicting,
            memory_only=memory_only,
            retained_evidence=memory_only.copy(),
            retained_outcome=retained_outcome,
            outcome_evidence=outcome_evidence,
            outcome_suppressed=outcome_suppressed,
        )

    def build_context(
        self, query: ParsedQuery, retrieved_memory_ids: list[str]
    ) -> MemoryContext:
        """Merge externally retrieved memories; unknown IDs raise KeyError.

        Task-mismatched outcomes expressed as evidence join the evidence merge.
        Query keys take priority. Multiple values within one memory are allowed;
        a key is withheld only when memories disagree on its value set.
        Identical evidence and outcomes appear once, in first-seen order by key
        and value.
        """
        query_keys = {unit.key for unit in query.evidence}
        evidence_by_key: dict[str, dict[str, EvidenceUnit]] = {}
        withheld_keys: set[str] = set()
        outcomes: dict[str, None] = {}

        for memory_id in retrieved_memory_ids:
            result = self.transform_memory(query, self.memories[memory_id])
            candidates = result.retained_evidence.copy()
            if result.outcome_evidence is not None:
                candidates.append(result.outcome_evidence)
            if result.retained_outcome is not None:
                outcomes[result.retained_outcome] = None

            memory_evidence_by_key: dict[str, dict[str, EvidenceUnit]] = {}
            for unit in candidates:
                if unit.key not in query_keys:
                    memory_evidence_by_key.setdefault(unit.key, {})[unit.value] = unit

            for key, values in memory_evidence_by_key.items():
                if key not in evidence_by_key:
                    evidence_by_key[key] = values
                elif evidence_by_key[key].keys() != values.keys():
                    withheld_keys.add(key)

        evidence = [
            unit
            for key, values in evidence_by_key.items()
            if key not in withheld_keys
            for unit in values.values()
        ]
        return MemoryContext(evidence=evidence, outcomes=list(outcomes))
