import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memtrim.core.schema import EvidenceUnit
from memtrim.core.trie import Trie


def unit(entity, attribute, value, value_type="free_text", multi_valued=False):
    return EvidenceUnit(entity=entity, attribute=attribute, value_type=value_type, value=value, span=value, multi_valued=multi_valued)


class TrieTest(unittest.TestCase):
    def test_exact_key_reused_across_memories(self):
        trie = Trie()
        u1 = unit("Northstar Robotics", "founding_year", "2016", value_type="numeric")
        u2 = unit("Northstar Robotics", "founding_year", "2016", value_type="numeric")
        trie.insert(u1, "m1", True)
        trie.insert(u2, "m2", True)
        self.assertEqual(u1.key_id, u2.key_id)
        self.assertEqual(u1.value_id, u2.value_id)
        self.assertEqual(trie.memory_ids_for(u1.key_id, u1.value_id), {"m1", "m2"})

    def test_different_values_branch_separately(self):
        trie = Trie()
        u1 = unit("Northstar Robotics", "founding_year", "2016", value_type="numeric")
        u2 = unit("Northstar Robotics", "founding_year", "2017", value_type="numeric")
        trie.insert(u1, "m1", True)
        trie.insert(u2, "m2", True)
        self.assertEqual(u1.key_id, u2.key_id)
        self.assertNotEqual(u1.value_id, u2.value_id)

    def test_multi_valued_flag_recorded(self):
        trie = Trie()
        u1 = unit("Atlas Foods", "founder", "Elena Cruz", multi_valued=True)
        trie.insert(u1, "m1", True)
        self.assertTrue(trie.is_multi_valued(u1.key_id))

    def test_query_match_is_read_only(self):
        trie = Trie()
        stored = unit("Atlas Foods", "founder", "Elena Cruz", multi_valued=True)
        trie.insert(stored, "m1", True)

        query_unit = unit("Atlas Foods", "founder", "Someone Else", multi_valued=True)
        key_id, value_id = trie.match_query_unit(query_unit)
        self.assertEqual(key_id, stored.key_id)
        self.assertIsNone(value_id)

        novel_entity_unit = unit("Nobody Inc", "founder", "Someone Else", multi_valued=True)
        key_id2, value_id2 = trie.match_query_unit(novel_entity_unit)
        self.assertIsNone(key_id2)
        self.assertIsNone(value_id2)

    def test_support_flag_tracked_per_memory(self):
        trie = Trie()
        u1 = unit("Northstar Robotics", "founding_year", "2016", value_type="numeric")
        u2 = unit("Northstar Robotics", "founding_year", "2016", value_type="numeric")
        trie.insert(u1, "m1", True)
        trie.insert(u2, "m2", False)
        leaf = trie.leaves[(u1.key_id, u1.value_id)]
        self.assertEqual(leaf["support_memory_ids"], {"m1"})
        self.assertEqual(leaf["memory_ids"], {"m1", "m2"})

    def test_task_canonicalization(self):
        trie = Trie()
        from memtrim.core.schema import TaskSignature

        t1 = TaskSignature(operation="founding_year", target="founding_year")
        t2 = TaskSignature(operation="Founding_Year", target="Founding_Year")
        trie.insert_task(t1)
        trie.insert_task(t2)
        self.assertEqual(t1.task_id, t2.task_id)


if __name__ == "__main__":
    unittest.main()
