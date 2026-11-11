import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memtrim.baselines import registry
from memtrim.core import prompts
from memtrim.datasets.base import Interaction, Item
from memtrim.experiments import main as runner
from memtrim.memory import registry as memory_registry
from memtrim.memory.base import MemoryBackend, make_entry


class FakeLLM:
    def __init__(self, reply="yes", parse=None):
        self.reply = reply
        self.parse = parse or {}
        self.calls = []

    def chat(self, messages, response_format=None, max_tokens=None, temperature=None, seed=None):
        self.calls.append(messages)
        if response_format is prompts.PARSE_SCHEMA:
            text = messages[-1]["content"].split("TEXT:\n", 1)[1]
            return {"content": json.dumps(self.parse[text])}
        if response_format is not None:
            return {"content": json.dumps({"equivalent": False, "candidate_index": -1})}
        return {"content": self.reply}


def fake_encode(texts):
    table = {"q": [1.0, 0.0], "a": [1.0, 0.05], "b": [1.0, 0.06], "c": [0.7, -0.7], "d": [0.0, 1.0]}

    def vec(t):
        v = np.asarray(table[t.split()[0]], dtype=float)
        return v / np.linalg.norm(v)

    if isinstance(texts, str):
        return vec(texts)
    return np.stack([vec(t) for t in texts])


def item(query="q current query", answer_type="yes_no"):
    return Item(id="i0", pair_id="p0", dataset="toy", condition="TD_DUP", group="g", query=query, gold="yes", answer_type=answer_type,
                memory={"text": "a stored", "answer": "no"}, store_id="s0")


def entries(texts, sources=None):
    out = []
    for i, t in enumerate(texts):
        e = make_entry(f"e{i}", t, f"ia{i}", 1.0 - i / 10)
        e["source_text"] = (sources or {}).get(i, f"source of {t}")
        out.append(e)
    return out


def user_content(llm, call=-1):
    return llm.calls[call][-1]["content"]


def system_content(llm, call=-1):
    return llm.calls[call][0]["content"]


class ContextTest(unittest.TestCase):
    def run_method(self, name, ents, llm=None, **params):
        llm = llm or FakeLLM()
        method = registry.create(name, top_k=params.pop("top_k", 5), **params)
        return method, method.run(item(), ents, llm), llm

    def test_no_memory_has_no_memory_block(self):
        _, result, llm = self.run_method("no_memory", entries(["a x", "b y"]))
        self.assertIsNone(result["context"])
        self.assertNotIn("MEMORY", user_content(llm))
        self.assertIn("q current query", user_content(llm))

    def test_base_memory_joins_top_k(self):
        _, result, llm = self.run_method("base_memory", entries(["a x", "b y", "c z"]), top_k=2)
        self.assertEqual(result["context"], "a x\nb y")
        self.assertIn("MEMORY:\na x\nb y", user_content(llm))

    def test_facts_only_uses_source_without_answer(self):
        ents = entries(["a x", "b y", "c z"], sources={0: "first fact question", 1: "first fact question", 2: None})
        ents[1]["interaction_id"] = ents[0]["interaction_id"]
        _, result, _ = self.run_method("facts_only", ents)
        self.assertEqual(result["context"], "first fact question")

    def test_top1(self):
        _, result, _ = self.run_method("top1", entries(["a x", "b y"]))
        self.assertEqual(result["context"], "a x")

    def test_mmr_prefers_diverse_entries(self):
        method, result, _ = self.run_method("mmr", entries(["a x", "b y", "c z", "d w"]), top_k=2, encode=fake_encode)
        self.assertEqual(method.pool_size, 15)
        self.assertEqual(result["context"], "a x\nc z")

    def test_sim_filter_threshold(self):
        _, result, _ = self.run_method("sim_filter", entries(["a x", "c z", "d w"]), encode=fake_encode)
        self.assertEqual(result["context"], "a x\nc z")

    def test_recomp_keeps_top_three_sentences(self):
        scores = {"S1.": 0.1, "S2.": 0.9, "S3.": 0.5, "S4.": 0.7}

        def scorer(query, sentences):
            return [scores[s] for s in sentences]

        _, result, _ = self.run_method("recomp", entries(["S1. S2.", "S3. S4."]), scorer=scorer)
        self.assertEqual(result["context"], "S2.\nS4.\nS3.")

    def test_llmlingua2_compresses_block_at_rate(self):
        seen = {}

        def compress(texts, rate):
            seen["texts"], seen["rate"] = texts, rate
            return " | ".join(t.split()[0] for t in texts)

        _, result, llm = self.run_method("llmlingua2", entries(["a x", "b y"]), compress=compress)
        self.assertEqual(seen, {"texts": ["a x", "b y"], "rate": 0.5})
        self.assertEqual(result["context"], "a | b")
        self.assertIn("MEMORY:\na | b", user_content(llm))

    def test_context_faithful_instruction(self):
        _, result, llm = self.run_method("context_faithful", entries(["a x"]))
        self.assertIn(prompts.CONTEXT_FAITHFUL_INSTRUCTION, system_content(llm))
        self.assertEqual(result["context"], "a x")

    def test_answer_first_two_calls(self):
        _, result, llm = self.run_method("answer_first", entries(["a x"]), llm=FakeLLM(reply="no"))
        self.assertEqual(len(llm.calls), 2)
        self.assertNotIn("MEMORY", user_content(llm, 0))
        self.assertIn("INITIAL ANSWER:\nno", user_content(llm, 1))
        self.assertIn(prompts.ANSWER_FIRST_INSTRUCTION, system_content(llm, 1))
        self.assertEqual(result["initial"], "no")

    def test_micro_act_lists_then_answers(self):
        _, result, llm = self.run_method("micro_act", entries(["a x"]))
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[0][0]["content"], prompts.MICRO_ACT_ANALYSIS_PROMPT)
        self.assertIn("COMPARISON:\nyes", user_content(llm, 1))
        self.assertIn(prompts.MICRO_ACT_INSTRUCTION, system_content(llm, 1))

    def test_same_template_across_methods(self):
        for name in ("base_memory", "top1", "facts_only", "context_faithful"):
            _, _, llm = self.run_method(name, entries(["a x"]))
            self.assertTrue(system_content(llm).startswith(prompts.ANSWER_SYSTEM_PROMPT))
            self.assertTrue(user_content(llm).startswith("CURRENT QUERY: q current query"))


def parse_json(entity, value, answer=None):
    unit = {"entity": entity, "attribute": "founding_year", "value_type": "numeric", "value": value, "span": f"founded in {value}", "multi_valued": False}
    return {
        "evidence": [unit], "task_operation": "founding_year", "task_target": "founding_year",
        "support_spans": [unit["span"]] if answer else [], "has_outcome": bool(answer), "outcome_text": answer or "",
        "outcome_statable_as_fact": False, "outcome_evidence_entity": "", "outcome_evidence_attribute": "",
        "outcome_evidence_value_type": "", "outcome_evidence_value": "",
    }


MEMORY_TEXT = "Acme was founded in 2016. When was Acme founded? 2016."
QUERY_TEXT = "Acme was founded in 2017. When was Acme founded?"
PARSES = {MEMORY_TEXT: parse_json("Acme", "2016", "2016"), QUERY_TEXT: parse_json("Acme", "2017")}


class FakeBackend(MemoryBackend):
    name = "fake"
    closed = 0

    def __init__(self, store_id, **kwargs):
        super().__init__(store_id)
        self.rows = []

    def add(self, interaction_id, text, answer, item_id=None, group=None, is_primary=True):
        self.rows.append(make_entry(f"m{len(self.rows)}", self.content(text, answer), interaction_id))

    def search(self, query, k):
        return [dict(r) for r in self.rows[:k]]

    def entries(self):
        return [dict(r) for r in self.rows]

    def close(self):
        FakeBackend.closed += 1


class MemtrimAndRunnerTest(unittest.TestCase):
    def test_memtrim_suppresses_answer_when_support_conflicts(self):
        llm = FakeLLM(parse=PARSES)
        backend = FakeBackend("s")
        backend.add("ia0", "Acme was founded in 2016. When was Acme founded?", "2016")
        method = registry.create("memtrim", embed_fn=None, workers=1).for_store(backend, llm)
        result = method.run(item(query=QUERY_TEXT, answer_type="entity"), backend.search(QUERY_TEXT, 5), llm)
        self.assertIsNone(result["context"])
        self.assertNotIn("MEMORY", user_content(llm))

    def test_runner_writes_every_method_once_and_resumes(self):
        original = memory_registry.create
        memory_registry.create = lambda name, store_id, **kw: FakeBackend(store_id)
        try:
            it = item(query=QUERY_TEXT, answer_type="entity")
            it.gold = "2017"
            interactions = [Interaction("ia0", "i0", "g", "Acme was founded in 2016. When was Acme founded?", "2016")]
            methods = [registry.create("base_memory"), registry.create("facts_only"),
                       registry.create("mmr", encode=lambda t: np.ones(2) / np.sqrt(2) if isinstance(t, str) else np.ones((len(t), 2)) / np.sqrt(2)),
                       registry.create("memtrim", embed_fn=None, workers=1)]
            with tempfile.TemporaryDirectory() as d:
                writer = runner.JsonlWriter(os.path.join(d, "out.jsonl"))
                llm = FakeLLM(reply="2017", parse=PARSES)
                runner.run_store("toy", 1, "fake", {}, "s0", interactions, [it], methods, llm, writer, 5)
                with open(writer.path) as f:
                    rows = [json.loads(line) for line in f]
                self.assertEqual([r["method"] for r in rows], ["base_memory", "facts_only", "mmr", "memtrim"])
                self.assertTrue(all(r["correct"] for r in rows))
                by_method = {r["method"]: r for r in rows}
                self.assertEqual(by_method["base_memory"]["context"], "Acme was founded in 2016. When was Acme founded? 2016.")
                self.assertEqual(by_method["facts_only"]["context"], "Acme was founded in 2016. When was Acme founded?")
                self.assertIsNone(by_method["memtrim"]["context"])
                closed = FakeBackend.closed
                runner.run_store("toy", 1, "fake", {}, "s0", interactions, [it], methods, llm, runner.JsonlWriter(writer.path), 5)
                with open(writer.path) as f:
                    self.assertEqual(len(f.readlines()), 4)
                self.assertEqual(FakeBackend.closed, closed)
        finally:
            memory_registry.create = original


if __name__ == "__main__":
    unittest.main()
