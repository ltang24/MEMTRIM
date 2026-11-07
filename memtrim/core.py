"""Write-time indexing (§3.2) and read-time replicate-aware context
construction (§3.3, Eq. 1, App. A.1)."""

from __future__ import annotations

from typing import Callable, TypedDict

import numpy as np

from .schema import (
    EvidenceUnit,
    MemoryRecord,
    MemoryTransformResult,
    MemTrimOptions,
    ParsedQuery,
)
from .trie import KEY_SEP, EvidenceTrie


class MemoryContext(TypedDict):
    """Merged evidence and stored outcomes for the caller to render."""

    evidence: list[EvidenceUnit]
    outcomes: list[str]
    raw_texts: list[str]


class MemTrim:
    """Transform memories selected by an external memory retriever.

    ``encode`` (texts -> L2-normalised matrix) is only needed for the
    ``embedding_match`` ablation.
    """

    def __init__(
        self,
        options: MemTrimOptions | None = None,
        encode: Callable[[list[str]], np.ndarray] | None = None,
    ) -> None:
        self.options = options or MemTrimOptions()
        self.memories: dict[str, MemoryRecord] = {}
        self.trie = EvidenceTrie()
        self._encode = encode

    # ------------------------------------------------------------- write
    def add_memory(self, memory: MemoryRecord) -> None:
        """Cache a parsed memory and index each unit independently.

        Duplicate IDs raise ValueError to prevent stale index entries.
        """
        if memory.memory_id in self.memories:
            raise ValueError(f"Memory ID already exists: {memory.memory_id}")
        self.memories[memory.memory_id] = memory
        support = set(memory.support_evidence)
        for unit in memory.evidence:
            self.trie.insert(unit, memory.memory_id, in_support=unit in support)

    # ---------------------------------------------------------- partition
    def _partition_trie(
        self, query: ParsedQuery, memory: MemoryRecord, retrieved: set[str]
    ) -> tuple[list[EvidenceUnit], list[EvidenceUnit], list[EvidenceUnit]]:
        """S_m(q), D_m(q), U_m(q) via independent trie lookups of each query unit."""
        mid = memory.memory_id
        shared_pairs: set[tuple[str, str]] = set()
        conflict_keys: set[str] = set()
        for unit in query.evidence:
            if mid in (self.trie.exact_lookup(unit.key, unit.value) & retrieved):
                shared_pairs.add((unit.key, unit.value))
            if mid in (self.trie.memories_for_key(unit.key) & retrieved):
                conflict_keys.add(unit.key)
        shared, conflicting, memory_only = [], [], []
        for unit in memory.evidence:
            if (unit.key, unit.value) in shared_pairs:
                shared.append(unit)
            elif unit.key in conflict_keys:
                conflicting.append(unit)
            else:
                memory_only.append(unit)
        return shared, conflicting, memory_only

    def _partition_embedding(
        self, query: ParsedQuery, memory: MemoryRecord
    ) -> tuple[list[EvidenceUnit], list[EvidenceUnit], list[EvidenceUnit]]:
        """Ablation: replace trie matching with embedding similarity."""
        if self._encode is None:
            raise RuntimeError("embedding_match ablation needs an encoder")
        if not query.evidence or not memory.evidence:
            return [], [], list(memory.evidence)
        q_full = self._encode([_unit_text(u) for u in query.evidence])
        m_full = self._encode([_unit_text(u) for u in memory.evidence])
        q_key = self._encode([_key_text(u.key) for u in query.evidence])
        m_key = self._encode([_key_text(u.key) for u in memory.evidence])
        full_sim = m_full @ q_full.T
        key_sim = m_key @ q_key.T
        shared, conflicting, memory_only = [], [], []
        for i, unit in enumerate(memory.evidence):
            if full_sim[i].max() >= self.options.embedding_shared_threshold:
                shared.append(unit)
            elif key_sim[i].max() >= self.options.embedding_key_threshold:
                conflicting.append(unit)
            else:
                memory_only.append(unit)
        return shared, conflicting, memory_only

    # ---------------------------------------------------------- Eq. (1)
    def transform_memory(
        self,
        query: ParsedQuery,
        memory: MemoryRecord,
        retrieved: set[str] | None = None,
    ) -> MemoryTransformResult:
        opt = self.options
        retrieved = retrieved if retrieved is not None else {memory.memory_id}
        if opt.embedding_match:
            shared, conflicting, memory_only = self._partition_embedding(query, memory)
        else:
            shared, conflicting, memory_only = self._partition_trie(query, memory, retrieved)

        retained = list(memory_only)
        if opt.keep_shared:
            retained += shared
        if opt.keep_conflicts:
            retained += conflicting

        # D_m(q) ∩ H_m != ∅  -> suppress the stored outcome.
        support = set(memory.support_evidence)
        support_conflict = any(u in support for u in conflicting)
        outcome_suppressed = (support_conflict and not opt.keep_old_answer) or memory.parse_failed

        retained_outcome: str | None = None
        outcome_evidence: EvidenceUnit | None = None
        if not outcome_suppressed:
            same_task = memory.task_signature == query.task_signature
            if not memory.outcome:
                pass  # nothing stored to reuse
            elif same_task or opt.ignore_task_change:
                retained_outcome = memory.outcome
            else:
                outcome_evidence = memory.outcome_evidence  # Type(o_m), may be None

        return MemoryTransformResult(
            memory_id=memory.memory_id,
            shared=shared,
            conflicting=conflicting,
            memory_only=memory_only,
            retained_evidence=retained,
            retained_outcome=retained_outcome,
            outcome_evidence=outcome_evidence,
            outcome_suppressed=outcome_suppressed,
        )

    # ------------------------------------------------------------- merge
    def build_context(self, query: ParsedQuery, retrieved_memory_ids: list[str]) -> MemoryContext:
        """Merge memories returned by the base retriever (App. A.1).

        * Trie matches are restricted to ``retrieved_memory_ids``.
        * The current query takes precedence for any key it provides.
        * Identical retained evidence is emitted once.
        * Different value sets for the same key across memories, unresolved by
          the query, are withheld (or resolved by timestamp if enabled).
        * Several retained outcomes that disagree are withheld for the same
          reason; agreeing outcomes are emitted once.
        """
        opt = self.options
        retrieved = set(retrieved_memory_ids)
        query_keys = {u.key for u in query.evidence}
        bypass_query_precedence = opt.keep_shared or opt.keep_conflicts

        per_key: dict[str, list[tuple[float, dict[str, EvidenceUnit]]]] = {}
        key_order: list[str] = []
        outcomes: dict[str, None] = {}
        raw_texts: list[str] = []

        for rank, memory_id in enumerate(retrieved_memory_ids):
            memory = self.memories[memory_id]
            if memory.parse_failed:
                if opt.on_parse_failure == "raw_without_outcome" and memory.source_text:
                    raw_texts.append(memory.source_text)
                continue
            result = self.transform_memory(query, memory, retrieved)
            candidates = list(result.retained_evidence)
            if result.outcome_evidence is not None:
                candidates.append(result.outcome_evidence)
            if result.retained_outcome is not None:
                outcomes[result.retained_outcome] = None

            this_memory: dict[str, dict[str, EvidenceUnit]] = {}
            for unit in candidates:
                if unit.key in query_keys and not bypass_query_precedence:
                    continue
                this_memory.setdefault(unit.key, {})[unit.value] = unit
            ts = memory.timestamp if memory.timestamp is not None else float(-rank)
            for key, values in this_memory.items():
                if key not in per_key:
                    key_order.append(key)
                per_key.setdefault(key, []).append((ts, values))

        evidence: list[EvidenceUnit] = []
        for key in key_order:
            versions = per_key[key]
            value_sets = {frozenset(v) for _, v in versions}
            if len(value_sets) == 1:
                evidence.extend(versions[0][1].values())
            elif opt.resolve_conflicts_by_timestamp and all(
                self._has_ts(ts) for ts, _ in versions
            ):
                evidence.extend(max(versions, key=lambda x: x[0])[1].values())
            # else: unresolved -> withheld

        outcome_list = list(outcomes)
        if len({o.strip().lower() for o in outcome_list}) > 1:
            outcome_list = []
        return MemoryContext(evidence=evidence, outcomes=outcome_list, raw_texts=raw_texts)

    @staticmethod
    def _has_ts(ts: float) -> bool:
        return ts is not None and ts > 0


# ----------------------------------------------------------------- render
def _key_parts(key: str) -> tuple[str, str, str]:
    parts = key.split(KEY_SEP)
    while len(parts) < 3:
        parts.append("")
    return parts[0], parts[1], parts[2]


def _key_text(key: str) -> str:
    entity, attribute, _ = _key_parts(key)
    return f"{entity} {attribute}".replace("_", " ").strip()


def _unit_text(unit: EvidenceUnit) -> str:
    return f"{_key_text(unit.key)}: {unit.value}"


def render_context(context: MemoryContext, task_labels: dict[str, str] | None = None) -> str:
    """Turn a MemoryContext into the text block given to the LLM."""
    lines: list[str] = []
    for unit in context["evidence"]:
        entity, attribute, _ = _key_parts(unit.key)
        attr = attribute.replace("_", " ")
        if attr.startswith("previous result:"):
            lines.append(f"- [{attr}] {entity}: {unit.value}")
        else:
            lines.append(f"- {entity.replace('_', ' ')} | {attr}: {unit.value}")
    for outcome in context["outcomes"]:
        lines.append(f"- Previous answer to this same question: {outcome}")
    for raw in context["raw_texts"]:
        lines.append(f"- (unparsed memory) {raw}")
    return "\n".join(lines)
