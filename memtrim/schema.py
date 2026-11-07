"""Data structures for canonical evidence and memory transformations (§3.1)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvidenceUnit:
    """A canonical key-value pair e = (k, v).

    ``key`` is the canonical key (entity | attribute | value type, after
    alignment, App. A.1). ``value`` is the normalised value. ``span`` points to
    the source text the unit was extracted from; it is not part of identity.
    """

    key: str
    value: str
    span: str | None = field(default=None, compare=False, hash=False)


@dataclass
class MemoryRecord:
    """m = (E_m, tau_m, H_m, o_m) plus bookkeeping."""

    memory_id: str
    evidence: list[EvidenceUnit]
    task_signature: str
    support_evidence: set[EvidenceUnit]
    outcome: str
    # Type(o_m): the outcome as an evidence unit labelled by its task, or None
    # when it cannot be expressed faithfully as a key-value unit (App. A.1).
    outcome_evidence: EvidenceUnit | None = None
    source_text: str = ""
    timestamp: float | None = None
    parse_failed: bool = False


@dataclass
class ParsedQuery:
    """E_q and tau_q for the current query."""

    evidence: list[EvidenceUnit]
    task_signature: str
    source_text: str = ""
    parse_failed: bool = False


@dataclass
class MemoryTransformResult:
    """Evidence partitions S_m(q), D_m(q), U_m(q) and outcome handling (Eq. 1)."""

    memory_id: str
    shared: list[EvidenceUnit]
    conflicting: list[EvidenceUnit]
    memory_only: list[EvidenceUnit]
    retained_evidence: list[EvidenceUnit]
    retained_outcome: str | None
    outcome_evidence: EvidenceUnit | None
    outcome_suppressed: bool


@dataclass
class MemTrimOptions:
    """Read-time switches. All False = full MEMTRIM. Each True = one ablation
    from §4.2 / Figure 5."""

    keep_shared: bool = False  # "Keep shared"
    keep_conflicts: bool = False  # "Keep conflicts"
    keep_old_answer: bool = False  # "Keep old answer"
    ignore_task_change: bool = False  # "Ignore task change"
    embedding_match: bool = False  # "Embedding match" (replace trie matching)
    embedding_shared_threshold: float = 0.90
    embedding_key_threshold: float = 0.80
    # App. A.1: by default unresolved cross-memory conflicts are withheld; if
    # the base system provides trusted timestamps they may resolve them.
    resolve_conflicts_by_timestamp: bool = False
    # App. A.1: on parser failure the raw memory stays in the base store but
    # its outcome is never reused. "raw_without_outcome" passes x_m as text;
    # "drop" passes nothing for that memory.
    on_parse_failure: str = "raw_without_outcome"

    @classmethod
    def ablation(cls, name: str | None) -> "MemTrimOptions":
        if name in (None, "", "full"):
            return cls()
        if not hasattr(cls(), name):
            raise ValueError(f"Unknown ablation: {name}")
        return cls(**{name: True})
