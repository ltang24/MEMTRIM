import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memtrim.datasets import cladder
from memtrim.datasets.base import TA, TD_DUP

QUESTION = "Will husband increase the chance of ringing alarm?"


def record(qid, p_x, p_y_given_x, answer, rung=2, question=QUESTION, reasoning=None):
    info_text = (
        f"The overall probability of alarm set by husband is {round(p_x * 100)}%. "
        f"For husbands that don't set the alarm, the probability of ringing alarm is {round(p_y_given_x[0] * 100)}%. "
        f"For husbands that set the alarm, the probability of ringing alarm is {round(p_y_given_x[1] * 100)}%."
    )
    if reasoning is None:
        reasoning = {"step4": f"P(Y=1 | X=0) = {p_y_given_x[0]:.2f}\nP(Y=1 | X=1) = {p_y_given_x[1]:.2f}", "end": "x"}
    return {
        "question_id": qid,
        "given_info": info_text,
        "question": question,
        "answer": answer,
        "background": "Husband has a direct effect on alarm clock.",
        "meta": {
            "story_id": "alarm",
            "graph_id": "chain",
            "query_type": "ate",
            "rung": rung,
            "model_id": 0,
            "given_info": {"p(X)": p_x, "p(Y | X)": list(p_y_given_x)},
        },
        "reasoning": reasoning,
    }


class CladderPairingTest(unittest.TestCase):
    def setUp(self):
        self.source = record(1, 0.60, (0.30, 0.70), "yes")
        self.candidate = record(2, 0.60, (0.30, 0.20), "no")

    def test_valid_pair(self):
        self.assertTrue(cladder.is_pair(self.source, self.candidate))

    def test_question_must_be_identical(self):
        other = record(2, 0.60, (0.30, 0.20), "no", question="Will husband decrease the chance of ringing alarm?")
        self.assertFalse(cladder.is_pair(self.source, other))

    def test_needs_an_unchanged_value(self):
        other = record(2, 0.10, (0.90, 0.20), "no")
        self.assertFalse(cladder.is_pair(self.source, other))

    def test_needs_a_changed_value(self):
        other = record(2, 0.60, (0.30, 0.70), "yes")
        self.assertFalse(cladder.is_pair(self.source, other))

    def test_changed_value_must_appear_in_candidate_reasoning(self):
        other = record(2, 0.60, (0.30, 0.20), "no", reasoning={"step4": "P(Y=1 | X=0) = 0.30", "end": "x"})
        self.assertFalse(cladder.is_pair(self.source, other))

    def test_key_jaccard_threshold(self):
        other = copy.deepcopy(self.candidate)
        other["meta"]["given_info"] = {"p(X)": 0.60, "p(V2 | X)": [0.1, 0.2]}
        self.assertLess(cladder.key_jaccard(self.source["meta"]["given_info"], other["meta"]["given_info"]), 0.8)
        self.assertFalse(cladder.is_pair(self.source, other))

    def test_masked_text_similarity_threshold(self):
        other = copy.deepcopy(self.candidate)
        other["given_info"] = "Completely unrelated wording about 20 kittens and 30 umbrellas on a sunny day."
        self.assertLess(cladder.masked_similarity(self.source["given_info"], other["given_info"]), 0.85)
        self.assertFalse(cladder.is_pair(self.source, other))
        self.assertEqual(cladder.masked_similarity("p is 30%", "p is 70%"), 1.0)

    def test_list_given_info_never_pairs(self):
        a, b = copy.deepcopy(self.source), copy.deepcopy(self.candidate)
        a["meta"]["given_info"] = [["V2"], []]
        b["meta"]["given_info"] = [["V2"], []]
        self.assertFalse(cladder.is_pair(a, b))

    def test_pairs_only_within_group(self):
        far = record(3, 0.60, (0.30, 0.20), "no", rung=1)
        pairs = cladder.find_pairs([self.source, self.candidate, far])
        ids = {(s["question_id"], c["question_id"]) for _, s, c in pairs}
        self.assertIn((1, 2), ids)
        self.assertNotIn((1, 3), ids)
        self.assertNotIn((3, 1), ids)

    def test_condition_and_roles(self):
        td = cladder.make_item("g", self.source, self.candidate)
        self.assertEqual(td.condition, TD_DUP)
        self.assertEqual(td.memory["answer"], "yes")
        self.assertEqual(td.gold, "no")
        self.assertIn("70%", td.memory["text"])
        self.assertIn("20%", td.query)
        ta_candidate = record(4, 0.60, (0.30, 0.90), "yes")
        self.assertEqual(cladder.make_item("g", self.source, ta_candidate).condition, TA)


if __name__ == "__main__":
    unittest.main()
