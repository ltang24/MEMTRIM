PARSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "parse_result",
        "schema": {
            "type": "object",
            "properties": {
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string"},
                            "attribute": {"type": "string"},
                            "value_type": {
                                "type": "string",
                                "enum": ["boolean", "numeric", "temporal", "categorical", "free_text"],
                            },
                            "value": {"type": "string"},
                            "span": {"type": "string"},
                            "multi_valued": {"type": "boolean"},
                        },
                        "required": ["entity", "attribute", "value_type", "value", "span", "multi_valued"],
                    },
                },
                "task_operation": {"type": "string"},
                "task_target": {"type": "string"},
                "support_spans": {"type": "array", "items": {"type": "string"}},
                "has_outcome": {"type": "boolean"},
                "outcome_text": {"type": "string"},
                "outcome_statable_as_fact": {"type": "boolean"},
                "outcome_evidence_entity": {"type": "string"},
                "outcome_evidence_attribute": {"type": "string"},
                "outcome_evidence_value_type": {"type": "string"},
                "outcome_evidence_value": {"type": "string"},
            },
            "required": [
                "evidence",
                "task_operation",
                "task_target",
                "support_spans",
                "has_outcome",
                "outcome_text",
                "outcome_statable_as_fact",
                "outcome_evidence_entity",
                "outcome_evidence_attribute",
                "outcome_evidence_value_type",
                "outcome_evidence_value",
            ],
        },
    },
}

PARSE_SYSTEM_PROMPT = """You extract structured evidence from one stored interaction or current query for a memory system called MEMTRIM.
Identify every atomic fact stated in the TEXT as an evidence unit (entity, attribute, value_type, value, span).
Rules:
- Produce one evidence unit per distinct (entity, attribute, value) fact literally stated in TEXT. Every person, number or name mentioned as filling the same role (e.g. every founder, every member) gets its own separate evidence unit, even if several appear in one list.
- Do not create an evidence unit for something TEXT only asks about without stating its value (a dangling question, e.g. "... is associated with is" with nothing after it). Only emit an evidence unit when TEXT actually states the value.
- span must be copied verbatim, character-for-character, from TEXT (a contiguous substring), proving where the fact came from. Use the shortest span that still proves the fact.
- value_type is one of boolean, numeric, temporal, categorical, free_text, chosen by the kind of value.
- attribute must be a specific noun phrase naming the kind of value being stated (e.g. country, founding_year, ceo, capital), not a vague verb or preposition phrase (e.g. avoid "associated_with", "related_to", "connected_to"); two texts stating the same kind of fact about an entity should end up with the same attribute wording.
- multi_valued is true if the attribute can naturally hold several simultaneous values for one entity (e.g. founder, member), false if the attribute normally has one value at a time (e.g. founding year, current capital).
- task_operation is a short label for what the TEXT is asking or asserting to determine (e.g. founding_year, route_planning). task_target further specifies the relation/aspect being asked about, written generically without repeating the entity name (e.g. founding_year, founders, route).
- support_spans lists the verbatim spans (subset of the spans used above) that the ANSWER (if any) directly depends on. If an evidence unit states the same fact as the answer itself (the TEXT is simply asserting a fact with no separate question), that evidence unit's span always belongs in support_spans. If you are unsure whether a span supports the answer, include it. If TEXT has no explicit answer, leave it empty.
- has_outcome is true only if TEXT states an explicit answer/conclusion (not just a question). outcome_text is that literal answer text, or empty string if none.
- outcome_statable_as_fact is true if the answer can be written as one key-value fact without changing its meaning (set outcome_evidence_* to that fact); false for procedural, subjective, or non-factual answers (leave outcome_evidence_* empty strings).
Return only the JSON object matching the schema.

Example TEXT:
Atlas Foods was founded in 2003 by Elena Cruz and Omar Said. When was Atlas Foods founded? Answer: 2003.

Example JSON:
{"evidence": [
  {"entity": "Atlas Foods", "attribute": "founding_year", "value_type": "numeric", "value": "2003", "span": "founded in 2003", "multi_valued": false},
  {"entity": "Atlas Foods", "attribute": "founder", "value_type": "free_text", "value": "Elena Cruz", "span": "Elena Cruz", "multi_valued": true},
  {"entity": "Atlas Foods", "attribute": "founder", "value_type": "free_text", "value": "Omar Said", "span": "Omar Said", "multi_valued": true}
], "task_operation": "founding_year", "task_target": "founding_year", "support_spans": ["founded in 2003"], "has_outcome": true, "outcome_text": "2003", "outcome_statable_as_fact": true, "outcome_evidence_entity": "Atlas Foods", "outcome_evidence_attribute": "founding_year", "outcome_evidence_value_type": "numeric", "outcome_evidence_value": "2003"}"""


def build_parse_messages(text, retry_error=None):
    content = f"TEXT:\n{text}"
    if retry_error:
        content += (
            f"\n\nYour previous attempt was rejected: {retry_error}\n"
            "Re-extract from the same TEXT, fixing this issue. Every span must be an exact verbatim substring of TEXT."
        )
    return [
        {"role": "system", "content": PARSE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


KEY_EQUIV_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "key_equivalence",
        "schema": {
            "type": "object",
            "properties": {
                "equivalent": {"type": "boolean"},
                "candidate_index": {"type": "integer"},
            },
            "required": ["equivalent", "candidate_index"],
        },
    },
}

KEY_EQUIV_SYSTEM_PROMPT = """You decide whether a newly observed entity/attribute key refers to the same underlying relation as one of several existing keys already indexed by a memory system.
Two keys are equivalent only if they denote the same attribute of the same entity, possibly phrased differently (e.g. "founded-in" and "founding year" of the same company).
Keys about a different attribute, or about a different entity, are not equivalent, even if the wording overlaps.
If unsure, answer false. If equivalent, candidate_index is the 0-based index of the matching existing key; otherwise candidate_index is -1."""


def build_key_equiv_messages(new_entity, new_attribute, new_value_type, candidates):
    lines = [f"New key: entity={new_entity!r}, attribute={new_attribute!r}, value_type={new_value_type!r}", "Existing keys:"]
    for i, c in enumerate(candidates):
        lines.append(f"[{i}] entity={c['entity']!r}, attribute={c['attribute']!r}, value_type={c['value_type']!r}")
    return [
        {"role": "system", "content": KEY_EQUIV_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


VALUE_EQUIV_SCHEMA = KEY_EQUIV_SCHEMA

VALUE_EQUIV_SYSTEM_PROMPT = """You decide whether a newly observed free-text value is equivalent to one of several existing values already stored under the same attribute key of a memory system.
Values are equivalent only if they state the same fact in different words (e.g. "CEO" and "Chief Executive Officer").
If unsure, answer false. If equivalent, candidate_index is the 0-based index of the matching existing value; otherwise candidate_index is -1."""


def build_value_equiv_messages(key_label, new_value, candidates):
    lines = [f"Attribute key: {key_label}", f"New value: {new_value!r}", "Existing values:"]
    for i, c in enumerate(candidates):
        lines.append(f"[{i}] {c!r}")
    return [
        {"role": "system", "content": VALUE_EQUIV_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


TASK_EQUIV_SYSTEM_PROMPT = """You decide whether two task signatures describe the same underlying question being asked, possibly about different entities.
Two task signatures are equivalent only if they ask the same kind of question (same operation and same target aspect), regardless of which entity they are about.
If unsure, answer false. If equivalent, candidate_index is the 0-based index of the matching existing task; otherwise candidate_index is -1."""


def build_task_equiv_messages(new_operation, new_target, candidates):
    lines = [f"New task: operation={new_operation!r}, target={new_target!r}", "Existing tasks:"]
    for i, c in enumerate(candidates):
        lines.append(f"[{i}] operation={c['operation']!r}, target={c['target']!r}")
    return [
        {"role": "system", "content": TASK_EQUIV_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


ANSWER_SYSTEM_PROMPT = """Answer the CURRENT QUESTION using the CURRENT FACT and, if present, the MEMORY block.
The MEMORY block may contain information from earlier, unrelated interactions. Use it only to the extent it actually helps answer the CURRENT QUESTION about the CURRENT FACT; do not let an old or differently-scoped memory override information explicitly given in the CURRENT FACT.
Give a short, direct answer with no explanation."""


def build_answer_messages(current_fact, question, memory_block=None):
    parts = [f"CURRENT FACT: {current_fact}", f"CURRENT QUESTION: {question}"]
    if memory_block:
        parts.append(f"MEMORY:\n{memory_block}")
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
