import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memtrim.datasets import counterlogic
from memtrim.datasets.base import TA, TD_DUP, compose


def record(rid, validity, believability, premises=("If A then B", "A"), conclusion="B"):
    return {"id": rid, "premise_list": list(premises), "conclusion": conclusion, "validity": validity, "believability": believability}


def fake_embed(texts):
    out = []
    for t in texts:
        if "daisies" in t:
            out.append([1.0, 0.0, 0.0])
        elif "dogs" in t:
            out.append([0.0, 1.0, 0.0])
        else:
            out.append([0.0, 0.0, 1.0])
    return np.asarray(out)


class CounterLogicTest(unittest.TestCase):
    def test_condition_from_validity_and_believability(self):
        self.assertEqual(counterlogic.condition_of(record("a", True, True)), TA)
        self.assertEqual(counterlogic.condition_of(record("b", False, False)), TA)
        self.assertEqual(counterlogic.condition_of(record("c", True, False)), TD_DUP)
        self.assertEqual(counterlogic.condition_of(record("d", False, True)), TD_DUP)

    def test_memory_asks_validity_and_query_asks_believability(self):
        item = counterlogic.make_item(record("x", True, False, ("If cats are dogs then dogs bark", "Cats are dogs"), "Dogs bark"), "c0")
        self.assertEqual(item.condition, TD_DUP)
        self.assertEqual(item.memory["answer"], "yes")
        self.assertEqual(item.gold, "no")
        self.assertEqual(item.answer_type, "yes_no")
        self.assertTrue(item.memory["text"].endswith(counterlogic.MEMORY_QUESTION))
        self.assertTrue(item.query.endswith(counterlogic.QUERY_QUESTION))
        argument = "Premises: If cats are dogs then dogs bark. Cats are dogs. Conclusion: Dogs bark."
        self.assertTrue(item.memory["text"].startswith(argument))
        self.assertTrue(item.query.startswith(argument))
        self.assertNotIn("yes", item.query.lower().split())
        self.assertEqual(compose(item.memory["text"], item.memory["answer"]), f"{argument} {counterlogic.MEMORY_QUESTION} yes.")

    def test_clusters_smaller_than_min_size_are_dropped(self):
        texts = ["daisies"] * 6 + ["dogs"] * 5 + ["warships"] * 4
        groups = counterlogic.cluster_groups(texts, fake_embed, distance_threshold=0.2, min_cluster_size=5)
        self.assertEqual(len(set(groups[:6])), 1)
        self.assertEqual(len(set(groups[6:11])), 1)
        self.assertNotEqual(groups[0], groups[6])
        self.assertTrue(all(g is None for g in groups[11:]))

    def test_build_samples_per_condition_and_caps_store_size(self):
        data = {"1": [], "2": []}
        for i in range(30):
            data["1"].append({"premise_list": ["Some daisies are flowers"], "conclusion": f"daisies {i}", "validity": i % 2 == 0, "believability": True, "premise": "", "num_additions": 0})
        for i in range(3):
            data["2"].append({"premise_list": ["Some warships float"], "conclusion": f"warships {i}", "validity": True, "believability": True, "premise": "", "num_additions": 0})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "data_v3.json")
            with open(path, "w") as f:
                json.dump(data, f)
            items, stores = counterlogic.build(path=path, n=10, seed=1, max_store_size=4, embed_fn=fake_embed)
        self.assertEqual(sum(1 for it in items if it.condition == TA), 10)
        self.assertEqual(sum(1 for it in items if it.condition == TD_DUP), 10)
        self.assertTrue(all("warships" not in it.query for it in items))
        self.assertTrue(all(len(v) <= 4 for v in stores.values()))
        for it in items:
            self.assertIn(f"{it.id}_primary", [ia.interaction_id for ia in stores[it.store_id]])
            self.assertTrue(it.store_id.startswith(it.group))


if __name__ == "__main__":
    unittest.main()
