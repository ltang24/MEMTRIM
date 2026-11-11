import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memtrim.core import parser


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, response_format=None, max_tokens=None, temperature=None, seed=None):
        content = self.responses[self.calls]
        self.calls += 1
        return {"content": content}


def good_response():
    return json.dumps(
        {
            "evidence": [
                {
                    "entity": "Atlas Foods",
                    "attribute": "founding_year",
                    "value_type": "numeric",
                    "value": "2003",
                    "span": "founded in 2003",
                    "multi_valued": False,
                }
            ],
            "task_operation": "founding_year",
            "task_target": "founding_year",
            "support_spans": ["founded in 2003"],
            "has_outcome": True,
            "outcome_text": "2003",
            "outcome_statable_as_fact": True,
            "outcome_evidence_entity": "Atlas Foods",
            "outcome_evidence_attribute": "founding_year",
            "outcome_evidence_value_type": "numeric",
            "outcome_evidence_value": "2003",
        }
    )


class ParserTest(unittest.TestCase):
    def test_well_formed_response_parses_on_first_try(self):
        client = FakeClient([good_response()])
        record = parser.parse_text(client, "Atlas Foods was founded in 2003.")
        self.assertTrue(record.parse_ok)
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(record.evidence), 1)
        self.assertEqual(record.evidence[0].value, "2003")
        self.assertEqual(record.outcome_text, "2003")

    def test_ungrounded_span_is_dropped_not_fatal(self):
        bad = json.loads(good_response())
        bad["evidence"].append(
            {
                "entity": "Atlas Foods",
                "attribute": "ceo",
                "value_type": "free_text",
                "value": "Nobody",
                "span": "text that does not appear anywhere",
                "multi_valued": False,
            }
        )
        client = FakeClient([json.dumps(bad)])
        record = parser.parse_text(client, "Atlas Foods was founded in 2003.")
        self.assertTrue(record.parse_ok)
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(record.evidence), 1)
        self.assertEqual(record.evidence[0].attribute, "founding_year")

    def test_missing_task_triggers_retry_then_succeeds(self):
        bad = json.loads(good_response())
        bad["task_operation"] = ""
        bad["task_target"] = ""
        client = FakeClient([json.dumps(bad), good_response()])
        record = parser.parse_text(client, "Atlas Foods was founded in 2003.")
        self.assertTrue(record.parse_ok)
        self.assertEqual(client.calls, 2)

    def test_still_broken_after_retry_falls_back_to_unparsed(self):
        bad = json.loads(good_response())
        bad["task_operation"] = ""
        client = FakeClient([json.dumps(bad), json.dumps(bad)])
        record = parser.parse_text(client, "Atlas Foods was founded in 2003.")
        self.assertFalse(record.parse_ok)
        self.assertEqual(client.calls, 2)
        self.assertEqual(record.evidence, [])

    def test_invalid_json_triggers_retry(self):
        client = FakeClient(["not json at all", good_response()])
        record = parser.parse_text(client, "Atlas Foods was founded in 2003.")
        self.assertTrue(record.parse_ok)
        self.assertEqual(client.calls, 2)

    def test_outcome_without_text_is_structural_error(self):
        bad = json.loads(good_response())
        bad["has_outcome"] = True
        bad["outcome_text"] = ""
        client = FakeClient([json.dumps(bad), json.dumps(bad)])
        record = parser.parse_text(client, "Atlas Foods was founded in 2003.")
        self.assertFalse(record.parse_ok)


if __name__ == "__main__":
    unittest.main()
