import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memtrim.memtrim import AblationFlags, build_memtrim_block, build_query_context, classify_memory, merge_contexts, transform_memory
from memtrim.schema import EvidenceUnit, ParsedRecord, TaskSignature
from memtrim.trie import Trie


def unit(entity, attribute, value, value_type="free_text", multi_valued=False):
    return EvidenceUnit(entity=entity, attribute=attribute, value_type=value_type, value=value, span=value, multi_valued=multi_valued)


def write_record(trie, record):
    for i, u in enumerate(record.evidence):
        trie.insert(u, id(record), record.support_mask[i])
    if record.outcome_evidence is not None:
        trie.insert(record.outcome_evidence, id(record), False)
    if record.task is not None:
        trie.insert_task(record.task)
    return record


class Eq1Test(unittest.TestCase):
    def setUp(self):
        self.trie = Trie()
        self.flags = AblationFlags()

    def _run_single_memory(self, memory_record, query_record):
        write_record(self.trie, memory_record)
        query_key_values, query_keys_present, query_task_id = build_query_context(query_record, self.trie)
        classifications = classify_memory(memory_record, query_record, query_key_values, query_keys_present, self.trie, self.flags)
        kept_units, outcome = transform_memory(memory_record, classifications, query_task_id, self.flags)
        return classifications, kept_units, outcome

    def test_support_conflict_suppresses_answer(self):
        founding_year_unit = unit("Northstar Robotics", "founding_year", "2016", value_type="numeric")
        memory = ParsedRecord(
            source_text="m",
            evidence=[founding_year_unit],
            task=TaskSignature(operation="founding_year", target="founding_year"),
            support_mask=[True],
            outcome_text="2016",
            outcome_evidence=unit("Northstar Robotics", "founding_year", "2016", value_type="numeric"),
        )
        query = ParsedRecord(
            source_text="q",
            evidence=[unit("Northstar Robotics", "founding_year", "2017", value_type="numeric")],
            task=TaskSignature(operation="founding_year", target="founding_year"),
            support_mask=[False],
        )
        classifications, kept_units, outcome = self._run_single_memory(memory, query)
        self.assertEqual(classifications, ["conflicting"])
        self.assertIsNone(outcome)
        self.assertEqual(kept_units, [])

    def test_unchanged_task_keeps_answer(self):
        memory = ParsedRecord(
            source_text="m",
            evidence=[unit("Northstar Robotics", "founding_year", "2016", value_type="numeric")],
            task=TaskSignature(operation="founding_year", target="founding_year"),
            support_mask=[True],
            outcome_text="2016",
            outcome_evidence=unit("Northstar Robotics", "founding_year", "2016", value_type="numeric"),
        )
        query = ParsedRecord(
            source_text="q",
            evidence=[],
            task=TaskSignature(operation="founding_year", target="founding_year"),
            support_mask=[],
        )
        _, kept_units, outcome = self._run_single_memory(memory, query)
        self.assertEqual(outcome, ("answer", "2016"))

    def test_task_change_relabels_outcome_as_evidence(self):
        memory = ParsedRecord(
            source_text="m",
            evidence=[
                unit("conclusion", "logical_validity", "valid", value_type="categorical"),
            ],
            task=TaskSignature(operation="validity", target="validity"),
            support_mask=[True],
            outcome_text="Yes",
            outcome_evidence=unit("conclusion", "validity", "yes", value_type="categorical"),
        )
        query = ParsedRecord(
            source_text="q",
            evidence=[],
            task=TaskSignature(operation="real_world_truth", target="real_world_truth"),
            support_mask=[],
        )
        _, kept_units, outcome = self._run_single_memory(memory, query)
        self.assertEqual(outcome[0], "labeled_fact")
        self.assertEqual(outcome[1].value, "yes")
        self.assertEqual(outcome[2].operation, "validity")

    def test_multi_valued_partial_overlap_is_memory_only_not_conflicting(self):
        memory = ParsedRecord(
            source_text="m",
            evidence=[
                unit("Acme Corp", "founder", "Maya Lee", multi_valued=True),
                unit("Acme Corp", "founder", "Daniel Ortiz", multi_valued=True),
            ],
            task=TaskSignature(operation="founders", target="founders"),
            support_mask=[True, True],
            outcome_text="Maya Lee and Daniel Ortiz",
            outcome_evidence=None,
        )
        query = ParsedRecord(
            source_text="q",
            evidence=[unit("Acme Corp", "founder", "Maya Lee", multi_valued=True)],
            task=TaskSignature(operation="founders", target="founders"),
            support_mask=[],
        )
        classifications, kept_units, outcome = self._run_single_memory(memory, query)
        self.assertEqual(classifications, ["shared", "memory_only"])
        self.assertEqual(len(kept_units), 1)
        self.assertEqual(kept_units[0].value, "Daniel Ortiz")
        self.assertEqual(outcome, ("answer", "Maya Lee and Daniel Ortiz"))

    def test_single_valued_conflict_suppresses_answer(self):
        memory = ParsedRecord(
            source_text="m",
            evidence=[unit("Acme Corp", "founding_year", "2016", value_type="numeric")],
            task=TaskSignature(operation="founding_year", target="founding_year"),
            support_mask=[True],
            outcome_text="2016",
            outcome_evidence=unit("Acme Corp", "founding_year", "2016", value_type="numeric"),
        )
        query = ParsedRecord(
            source_text="q",
            evidence=[unit("Acme Corp", "founding_year", "2017", value_type="numeric")],
            task=TaskSignature(operation="founding_year", target="founding_year"),
            support_mask=[],
        )
        classifications, kept_units, outcome = self._run_single_memory(memory, query)
        self.assertEqual(classifications, ["conflicting"])
        self.assertIsNone(outcome)

    def test_merge_across_memories_withholds_unresolved_single_valued_conflict(self):
        trie = Trie()
        m1 = ParsedRecord(
            source_text="m1",
            evidence=[unit("Acme Corp", "founding_year", "2016", value_type="numeric")],
            task=TaskSignature(operation="founding_year", target="founding_year"),
            support_mask=[False],
        )
        m2 = ParsedRecord(
            source_text="m2",
            evidence=[unit("Acme Corp", "founding_year", "2019", value_type="numeric")],
            task=TaskSignature(operation="founding_year", target="founding_year"),
            support_mask=[False],
        )
        write_record(trie, m1)
        write_record(trie, m2)
        query = ParsedRecord(source_text="q", evidence=[], task=TaskSignature(operation="other", target="other"), support_mask=[])
        query_key_values, query_keys_present, query_task_id = build_query_context(query, trie)
        flags = AblationFlags()
        results = []
        for m in (m1, m2):
            classifications = classify_memory(m, query, query_key_values, query_keys_present, trie, flags)
            kept_units, outcome = transform_memory(m, classifications, query_task_id, flags)
            results.append((kept_units, outcome))
        final_items, kept_answer = merge_contexts(results, trie)
        self.assertEqual(final_items, [])

    def test_merge_across_memories_combines_multi_valued(self):
        trie = Trie()
        m1 = ParsedRecord(
            source_text="m1",
            evidence=[unit("Acme Corp", "founder", "Maya Lee", multi_valued=True)],
            task=TaskSignature(operation="founders", target="founders"),
            support_mask=[False],
        )
        m2 = ParsedRecord(
            source_text="m2",
            evidence=[unit("Acme Corp", "founder", "Daniel Ortiz", multi_valued=True)],
            task=TaskSignature(operation="founders", target="founders"),
            support_mask=[False],
        )
        write_record(trie, m1)
        write_record(trie, m2)
        query = ParsedRecord(source_text="q", evidence=[], task=TaskSignature(operation="other", target="other"), support_mask=[])
        query_key_values, query_keys_present, query_task_id = build_query_context(query, trie)
        flags = AblationFlags()
        results = []
        for m in (m1, m2):
            classifications = classify_memory(m, query, query_key_values, query_keys_present, trie, flags)
            kept_units, outcome = transform_memory(m, classifications, query_task_id, flags)
            results.append((kept_units, outcome))
        final_items, kept_answer = merge_contexts(results, trie)
        values = sorted(u.value for u in final_items)
        self.assertEqual(values, ["Daniel Ortiz", "Maya Lee"])

    def test_merge_withholds_disagreeing_answers(self):
        trie = Trie()
        m1 = ParsedRecord(
            source_text="m1",
            evidence=[],
            task=TaskSignature(operation="founding_year", target="founding_year"),
            support_mask=[],
            outcome_text="2016",
        )
        m2 = ParsedRecord(
            source_text="m2",
            evidence=[],
            task=TaskSignature(operation="founding_year", target="founding_year"),
            support_mask=[],
            outcome_text="2017",
        )
        write_record(trie, m1)
        write_record(trie, m2)
        query = ParsedRecord(source_text="q", evidence=[], task=TaskSignature(operation="founding_year", target="founding_year"), support_mask=[])
        query_key_values, query_keys_present, query_task_id = build_query_context(query, trie)
        flags = AblationFlags()
        results = []
        for m in (m1, m2):
            classifications = classify_memory(m, query, query_key_values, query_keys_present, trie, flags)
            kept_units, outcome = transform_memory(m, classifications, query_task_id, flags)
            results.append((kept_units, outcome))
        final_items, kept_answer = merge_contexts(results, trie)
        self.assertIsNone(kept_answer)


if __name__ == "__main__":
    unittest.main()
