"""Focused regression checks for merging evidence across stored memories."""

import unittest

from memtrim import EvidenceUnit, MemTrim, MemoryRecord, ParsedQuery


class ContextMergeTests(unittest.TestCase):
    def merged_evidence(self, memories, query_evidence=()):
        memtrim = MemTrim()
        memory_ids = []
        for index, evidence in enumerate(memories):
            memory_id = f"m{index + 1}"
            memory_ids.append(memory_id)
            memtrim.add_memory(
                MemoryRecord(
                    memory_id=memory_id,
                    evidence=list(evidence),
                    task_signature="test",
                    support_evidence=set(),
                    outcome="stored outcome",
                )
            )
        query = ParsedQuery(evidence=list(query_evidence), task_signature="test")
        return memtrim.build_context(query, memory_ids)["evidence"]

    def test_one_memory_retains_multiple_values_in_order(self):
        founders = [
            EvidenceUnit("founder", "Maya Lee"),
            EvidenceUnit("founder", "Daniel Ortiz"),
        ]
        self.assertEqual(self.merged_evidence([founders]), founders)

    def test_identical_evidence_across_memories_is_emitted_once(self):
        year = EvidenceUnit("founded_in", "2016")
        self.assertEqual(self.merged_evidence([[year], [year]]), [year])

    def test_conflicting_memories_withhold_unresolved_key(self):
        self.assertEqual(
            self.merged_evidence(
                [[EvidenceUnit("status", "open")], [EvidenceUnit("status", "closed")]]
            ),
            [],
        )

    def test_equal_value_sets_keep_first_memory_order(self):
        founders = [
            EvidenceUnit("founder", "Maya Lee"),
            EvidenceUnit("founder", "Daniel Ortiz"),
        ]
        self.assertEqual(
            self.merged_evidence([founders, list(reversed(founders))]), founders
        )

    def test_partially_overlapping_value_sets_are_withheld(self):
        maya = EvidenceUnit("founder", "Maya Lee")
        daniel = EvidenceUnit("founder", "Daniel Ortiz")
        self.assertEqual(self.merged_evidence([[maya, daniel], [maya]]), [])

    def test_missing_key_in_another_memory_is_not_a_conflict(self):
        founder = EvidenceUnit("founder", "Maya Lee")
        status = EvidenceUnit("status", "open")
        self.assertEqual(self.merged_evidence([[founder], [status]]), [founder, status])

    def test_query_key_takes_precedence(self):
        founder = EvidenceUnit("founder", "Maya Lee")
        self.assertEqual(
            self.merged_evidence(
                [
                    [EvidenceUnit("status", "open"), founder],
                    [EvidenceUnit("status", "closed")],
                ],
                query_evidence=[EvidenceUnit("status", "open")],
            ),
            [founder],
        )


if __name__ == "__main__":
    unittest.main()
