
import json
import unittest

import numpy as np

from memtrim import EvidenceUnit as E
from memtrim import MemoryRecord, MemTrim, MemTrimOptions, ParsedQuery, render_context
from memtrim.experiments.rq3 import controlled_query
from memtrim.llm import LLM, LLMConfig
from memtrim.parser import EvidenceParser
from memtrim.stats import overlap_tests, quadratic, two_lines
from memtrim.trie import EvidenceTrie


def rec(mid, ev, task, support, outcome, oev=None):
    return MemoryRecord(mid, list(ev), task, set(support), outcome, oev)


class TrieTests(unittest.TestCase):
    def test_shared_prefix_and_value_branches(self):
        t = EvidenceTrie()
        t.insert(E("openai|founded_in|temporal", "2015"), "m1", True)
        t.insert(E("openai|founded_in|temporal", "2015"), "m2")
        t.insert(E("openai|founded_in|temporal", "2016"), "m3")
        t.insert(E("openai|founder|entity", "sam altman"), "m1")
        self.assertEqual(t.exact_lookup("openai|founded_in|temporal", "2015"), {"m1", "m2"})
        self.assertEqual(t.values_for_key("openai|founded_in|temporal"), {"2015", "2016"})
        self.assertEqual(len(t.root.children), 1)  # one shared 'openai' node
        self.assertTrue(t.is_support("openai|founded_in|temporal", "2015", "m1"))
        self.assertFalse(t.is_support("openai|founded_in|temporal", "2015", "m2"))


class Eq1Tests(unittest.TestCase):
    def setUp(self):
        self.club = E("lionel_messi|club|entity", "paris saint germain")
        self.person = E("question|subject|entity", "lionel messi")
        self.pos = E("lionel_messi|position|categorical", "forward")
        self.mem = rec("m", [self.person, self.club, self.pos], "identify:club_country",
                       [self.club], "France", E("lionel_messi|previous result: identify:club_country|entity", "france"))

    def _ctx(self, query, options=None):
        mt = MemTrim(options)
        mt.add_memory(self.mem)
        return mt.build_context(query, ["m"])

    def test_case1_support_conflict_suppresses_outcome(self):
        q = ParsedQuery([self.person, E("lionel_messi|club|entity", "inter miami")], "identify:club_country")
        ctx = self._ctx(q)
        self.assertEqual(ctx["outcomes"], [])
        self.assertEqual(ctx["evidence"], [self.pos])  # memory-only kept; shared & conflict removed

    def test_same_task_no_conflict_keeps_outcome(self):
        q = ParsedQuery([self.person, self.club], "identify:club_country")
        self.assertEqual(self._ctx(q)["outcomes"], ["France"])

    def test_task_change_converts_outcome_to_labelled_evidence(self):
        q = ParsedQuery([self.person, self.club], "identify:club_league")
        ctx = self._ctx(q)
        self.assertEqual(ctx["outcomes"], [])
        self.assertIn(self.mem.outcome_evidence, ctx["evidence"])
        self.assertIn("previous result", render_context(ctx))

    def test_ablations(self):
        q = ParsedQuery([self.person, E("lionel_messi|club|entity", "inter miami")], "identify:club_country")
        self.assertIn(self.person, self._ctx(q, MemTrimOptions(keep_shared=True))["evidence"])
        self.assertIn(self.club, self._ctx(q, MemTrimOptions(keep_conflicts=True))["evidence"])
        self.assertEqual(self._ctx(q, MemTrimOptions(keep_old_answer=True))["outcomes"], ["France"])
        q2 = ParsedQuery([self.person, self.club], "identify:club_league")
        self.assertEqual(self._ctx(q2, MemTrimOptions(ignore_task_change=True))["outcomes"], ["France"])

    def test_sparse_overlap(self):
        e1, e2, e3, e4 = (E(f"x|a{i}|t", "v") for i in range(4))
        mt = MemTrim()
        mt.add_memory(rec("m", [e1, e2, e3, e4], "t", [], "o"))
        q = ParsedQuery([e1, E("x|a1|t", "w"), e3, E("x|a3|t", "w")], "t")
        r = mt.transform_memory(q, mt.memories["m"])
        self.assertEqual(r.shared, [e1, e3])
        self.assertEqual(r.conflicting, [e2, e4])

    def test_disagreeing_outcomes_withheld(self):
        mt = MemTrim()
        mt.add_memory(rec("a", [], "t", [], "yes"))
        mt.add_memory(rec("b", [], "t", [], "no"))
        self.assertEqual(mt.build_context(ParsedQuery([], "t"), ["a", "b"])["outcomes"], [])

    def test_parse_failure_never_reuses_outcome(self):
        mt = MemTrim()
        m = rec("a", [], "", [], "yes")
        m.parse_failed, m.source_text = True, "raw text"
        mt.add_memory(m)
        ctx = mt.build_context(ParsedQuery([], "t"), ["a"])
        self.assertEqual(ctx["outcomes"], [])
        self.assertEqual(ctx["raw_texts"], ["raw text"])

    def test_timestamp_resolution_optional(self):
        k = "road|status|categorical"
        a, b = rec("a", [E(k, "open")], "t", [], ""), rec("b", [E(k, "closed")], "t", [], "")
        a.timestamp, b.timestamp = 1.0, 2.0
        for opts, expect in ((MemTrimOptions(), []), (MemTrimOptions(resolve_conflicts_by_timestamp=True), [E(k, "closed")])):
            mt = MemTrim(opts)
            mt.add_memory(a)
            mt.add_memory(b)
            self.assertEqual(mt.build_context(ParsedQuery([], "t"), ["a", "b"])["evidence"], expect)


class CaseStudies(unittest.TestCase):
    """App. B.4 with hand-written parses: the context MEMTRIM passes on."""

    def test_case2_counterlogic_task_change(self):
        p1 = E("metals|liquid_at_room_temperature|boolean", "true")
        p2 = E("iron|is_metal|boolean", "true")
        c = E("iron|liquid_at_room_temperature|boolean", "true")
        mem = rec("m", [p1, p2, c], "decide:logical_validity", [p1, p2, c], "Yes",
                  E("argument|previous result: decide:logical_validity|boolean", "true"))
        mt = MemTrim()
        mt.add_memory(mem)
        ctx = mt.build_context(ParsedQuery([p1, p2, c], "decide:real_world_truth"), ["m"])
        self.assertEqual(ctx["outcomes"], [])
        self.assertEqual(ctx["evidence"], [mem.outcome_evidence])  # only (validity, yes)

    def test_case3_cladder_premise_change(self):
        g = E("graph|structure|text", "z->x,z->y,x->y")
        p1, p0 = E("y|p_given_x1|numeric", "0.72"), E("y|p_given_x0|numeric", "0.41")
        mem = rec("m", [g, p1, p0], "decide:causal_effect", [g, p1, p0], "Yes")
        mt = MemTrim()
        mt.add_memory(mem)
        q = ParsedQuery([g, E("y|p_do_x1|numeric", "0.38"), E("y|p_given_x0|numeric", "0.40")], "decide:causal_effect")
        ctx = mt.build_context(q, ["m"])
        self.assertEqual(ctx["outcomes"], [])
        self.assertNotIn(p0, ctx["evidence"])


class _FakeLLM:
    """Returns queued strings; records prompts."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def chat(self, messages, **kw):
        self.prompts.append(messages)
        return self.outputs.pop(0) if self.outputs else json.dumps({"match": -1})


class ParserTests(unittest.TestCase):
    def test_grounding_support_retry_and_canonicalisation(self):
        good = {
            "evidence": [
                {"entity": "Highway 5", "attribute": "status", "value_type": "categorical", "value": "Closed", "span": "Highway 5 is closed"},
                {"entity": "Moon", "attribute": "material", "value_type": "text", "value": "cheese", "span": "the moon is cheese"},
                {"entity": "Trip", "attribute": "distance", "value_type": "numeric", "value": "12%", "span": "12%"},
            ],
            "task": {"operation": "recommend", "target": "route"},
            "support": [0],
            "answer_as_evidence": {"entity": "trip", "attribute": "route", "value_type": "entity", "value": "Highway 7"},
        }
        llm = _FakeLLM(["not json", json.dumps(good)])
        p = EvidenceParser(llm, encoder=None)
        r = p.parse_memory("m1", "Highway 5 is closed today. Distance up 12% this week. Which route?", "Highway 7")
        self.assertEqual(p.stats.n_retries, 1)
        self.assertEqual(p.stats.n_ungrounded_units, 1)  # 'the moon is cheese' not in input
        self.assertEqual([u.value for u in r.evidence], ["closed", "0.12"])
        self.assertEqual({u.value for u in r.support_evidence}, {"closed"})
        self.assertEqual(r.task_signature, "recommend:route")
        self.assertIn("previous result", r.outcome_evidence.key)
        # second key with the same entity + type triggers an equivalence check
        q = {"evidence": [{"entity": "Highway 5", "attribute": "state", "value_type": "categorical", "value": "open", "span": "Highway 5 is open"}],
             "task": {"operation": "recommend", "target": "route"}}
        llm.outputs = [json.dumps(q), json.dumps({"match": 0})]
        pq = p.parse_query("Highway 5 is open now. Which route?")
        self.assertEqual(pq.evidence[0].key, "highway_5|status|categorical")
        self.assertEqual(p.stats.n_key_merges, 1)

    def test_failure_after_retry(self):
        p = EvidenceParser(_FakeLLM(["x", "y"]), encoder=None)
        r = p.parse_memory("m", "text", "ans")
        self.assertTrue(r.parse_failed)

    def test_mock_backend_runs(self):
        llm = LLM(LLMConfig(backend="mock", model="mock", cache_dir=None))
        p = EvidenceParser(llm, encoder=None)
        r = p.parse_memory("m", "Context: Messi plays for PSG.\nQuestion: Which country?", "France")
        self.assertFalse(r.parse_failed)
        self.assertTrue(r.evidence)


class StatsTests(unittest.TestCase):
    def test_two_lines_detects_u(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, 600)
        y = (x - 0.5) ** 2 * 4 - 0.5 + rng.normal(0, 0.1, 600)
        self.assertTrue(two_lines(x, y, "u")["ok"])
        self.assertFalse(two_lines(x, x + rng.normal(0, 0.1, 600), "u")["ok"])
        self.assertLess(quadratic(x, y)["p"], 0.05)

    def test_overlap_centrality(self):
        rng = np.random.default_rng(1)
        succ = np.concatenate([rng.uniform(0, 0.2, 150), rng.uniform(0.8, 1, 150)])
        fail = rng.normal(0.5, 0.08, 200).clip(0, 1)
        t = overlap_tests(succ, fail)
        self.assertLess(t["centrality_p"], 0.05)
        self.assertTrue(t["two_lines"]["ok"])


class RQ3Tests(unittest.TestCase):
    item = {"id": "x", "controlled": {
        "units": [{"mem_text": "Old fact.", "td_text": "New fact.", "support": True},
                  {"mem_text": "A.", "td_text": None, "support": False},
                  {"mem_text": "B.", "td_text": None, "support": False}],
        "question_ta": "Q?", "question_td": "Q?", "ta_gold": "old", "ta_aliases": [],
        "td_gold": "new", "td_aliases": [], "answer_type": "entity"}}

    def test_levels(self):
        t0, g0, _, k0, n = controlled_query(self.item, 0.0, 0)
        self.assertEqual((g0, k0, n), ("new", 0, 3))
        self.assertIn("New fact.", t0)
        self.assertNotIn("A.", t0)
        t_mid, g_mid, *_ = controlled_query(self.item, 0.67, 0)
        self.assertEqual(g_mid, "new")
        self.assertIn("A.", t_mid)
        self.assertIn("B.", t_mid)
        t1, g1, *_ = controlled_query(self.item, 1.0, 0)
        self.assertEqual(g1, "old")
        self.assertIn("Old fact.", t1)
        self.assertNotIn("New fact.", t1)


if __name__ == "__main__":
    unittest.main()
