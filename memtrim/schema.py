"""Data structures for canonical evidence and memory transformations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceUnit:
    """A canonical key-value pair."""

    key: str
    value: str


@dataclass
class MemoryRecord:
    """Stored evidence, task, and the evidence subset supporting its outcome."""

    memory_id: str
    evidence: list[EvidenceUnit]
    task_signature: str
    support_evidence: set[EvidenceUnit]
    outcome: str
    outcome_evidence: EvidenceUnit | None = None


@dataclass
class ParsedQuery:
    """Already-parsed evidence and the current task signature."""

    evidence: list[EvidenceUnit]
    task_signature: str


@dataclass
class MemoryTransformResult:
    """Evidence partitions and the separately handled stored outcome."""

    memory_id: str
    shared: list[EvidenceUnit]
    conflicting: list[EvidenceUnit]
    memory_only: list[EvidenceUnit]
    retained_evidence: list[EvidenceUnit]
    retained_outcome: str | None
    outcome_evidence: EvidenceUnit | None
    outcome_suppressed: bool
