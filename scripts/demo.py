"""Demonstrate matching and conflicting queries with deterministic toy evidence."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution with: python scripts/demo.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memtrim import EvidenceUnit, MemoryRecord, MemTrim, ParsedQuery


def print_evidence(label: str, evidence: list[EvidenceUnit]) -> None:
    """Print canonical evidence pairs in their existing order."""
    print(f"{label}:")
    if not evidence:
        print("(none)")
    for unit in evidence:
        print(f"- {unit.key} = {unit.value}")


def main() -> None:
    memory = MemoryRecord(
        memory_id="northstar-memory",
        evidence=[
            EvidenceUnit("company", "Northstar Robotics"),
            EvidenceUnit("founded_in", "2016"),
            EvidenceUnit("founder", "Maya Lee"),
            EvidenceUnit("founder", "Daniel Ortiz"),
        ],
        task_signature="founding_year",
        support_evidence={EvidenceUnit("founded_in", "2016")},
        outcome="2016",
    )
    memtrim = MemTrim()
    memtrim.add_memory(memory)

    for label, year in [("MATCH", "2016"), ("MISMATCH", "2017")]:
        if label == "MISMATCH":
            print()
        print(f"=== {label} CASE ===")
        query = ParsedQuery(
            evidence=[
                EvidenceUnit("company", "Northstar Robotics"),
                EvidenceUnit("founded_in", year),
            ],
            task_signature="founding_year",
        )
        result = memtrim.transform_memory(query, memory)
        print_evidence("Shared", result.shared)
        print_evidence("Conflicting", result.conflicting)
        print_evidence("Memory-only", result.memory_only)
        print(f"Retained outcome: {result.retained_outcome}")
        print(f"Outcome suppressed: {result.outcome_suppressed}")

        context = memtrim.build_context(query, [memory.memory_id])
        print("Final memory context:")
        # Both founder values come from one memory and survive the merge.
        print_evidence("Evidence", context["evidence"])
        print("Outcomes:")
        if not context["outcomes"]:
            print("(none)")
        for outcome in context["outcomes"]:
            print(f"- {outcome}")


if __name__ == "__main__":
    main()
