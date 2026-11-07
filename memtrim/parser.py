
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from .embed import Encoder
from .llm import LLM, extract_json
from .schema import EvidenceUnit, MemoryRecord, ParsedQuery
from .text import normalize
from .trie import KEY_SEP

VALUE_TYPES = ("boolean", "categorical", "numeric", "temporal", "entity", "text")

_PARSE_SYSTEM = """You convert an interaction into atomic, typed evidence records. <<MOCK:{kind}>>
Return ONLY a JSON object with the fields described below. Rules:
- An evidence record states ONE atomic fact given in the input: the entity it
  describes, the attribute or relation, the value, and the value type
  (one of: boolean, categorical, numeric, temporal, entity, text).
- Use a short canonical noun phrase for the entity and a snake_case attribute,
  e.g. "Highway 5 is closed" -> entity "Highway 5", attribute "status",
  value "closed", value_type "categorical".
- Express a probability or percentage as a number, e.g. "8%" -> "0.08".
- "span" must be copied VERBATIM from the input text it came from.
- Do not invent facts that are not in the input. Do not include the requested
  answer itself as an evidence record.
- "task" describes what the question asks the model to determine:
  "operation" (e.g. identify, decide_yes_no, recommend, compute) and
  "target" (the attribute or quantity asked about, WITHOUT the specific entity
  names, e.g. "club_country", "logical_validity", "real_world_truth",
  "causal_effect_sign", "founding_year").
{memory_rules}"""

_MEMORY_RULES = """- The input also contains the stored ANSWER. List in "support" the indices
  (0-based, into "evidence") of every record the answer depends on. If you are
  unsure whether the answer depends on a record, INCLUDE it.
- "answer_as_evidence": if the answer can be stated as a factual key-value
  record without changing its meaning, give it as
  {"entity", "attribute", "value_type", "value"}; for procedural, subjective
  or otherwise non-factual answers use null.
Output schema:
{"evidence": [{"entity": str, "attribute": str, "value_type": str, "value": str, "span": str}],
 "task": {"operation": str, "target": str},
 "support": [int],
 "answer_as_evidence": {...} | null}"""

_QUERY_RULES = """Output schema:
{"evidence": [{"entity": str, "attribute": str, "value_type": str, "value": str, "span": str}],
 "task": {"operation": str, "target": str}}"""

_EQUIV_PROMPT = """<<MOCK:{kind}>>You decide whether a NEW {what} means exactly the same thing as one of the
EXISTING ones. Two keys are equivalent only if they describe the same attribute
of the same entity (different wording or aliases are fine). Similar wording
alone is NOT enough. If you are not confident, answer -1.

NEW: {new}
EXISTING:
{existing}

Return ONLY a JSON object: {{"match": <index of the equivalent EXISTING item, or -1>}}"""


@dataclass
class ParserStats:
    n_parsed: int = 0
    n_retries: int = 0
    n_failures: int = 0
    n_ungrounded_units: int = 0
    n_key_merges: int = 0
    n_task_merges: int = 0


@dataclass
class _Registry:
    """Canonical keys / task signatures seen so far."""

    items: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    alias: dict[str, str] = field(default_factory=dict)  # raw -> canonical


class EvidenceParser:
    def __init__(
        self,
        llm: LLM,
        encoder: Encoder | None = None,
        n_candidates: int = 5,
        candidate_sim: float = 0.6,
        max_retries: int = 1,
    ) -> None:
        self.llm = llm
        self.encoder = encoder
        self.n_candidates = n_candidates
        self.candidate_sim = candidate_sim
        self.max_retries = max_retries
        self.keys = _Registry()
        self.tasks = _Registry()
        self.values: dict[str, list[str]] = {}  # canonical key -> free-text values
        self.stats = ParserStats()

    # ------------------------------------------------------------ public
    def parse_memory(
        self, memory_id: str, interaction: str, answer: str, timestamp: float | None = None
    ) -> MemoryRecord:
        text = f"{interaction}\nANSWER: {answer}"
        raw = self._call_parser(text, kind="parse_memory", ground_text=interaction)
        if raw is None:
            self.stats.n_failures += 1
            return MemoryRecord(
                memory_id=memory_id,
                evidence=[],
                task_signature="",
                support_evidence=set(),
                outcome=answer,
                outcome_evidence=None,
                source_text=interaction,
                timestamp=timestamp,
                parse_failed=True,
            )
        units, kept_idx = self._units(raw["evidence"], interaction)
        support_idx = set(int(i) for i in raw.get("support", []) if isinstance(i, (int, float)))
        if not support_idx:
            # Conservative policy: unknown dependence -> everything supports.
            support = set(units)
        else:
            support = {u for u, i in zip(units, kept_idx) if i in support_idx}
        task = self._task(raw["task"])
        outcome_ev = self._outcome_evidence(raw.get("answer_as_evidence"), task)
        return MemoryRecord(
            memory_id=memory_id,
            evidence=units,
            task_signature=task,
            support_evidence=support,
            outcome=answer,
            outcome_evidence=outcome_ev,
            source_text=interaction,
            timestamp=timestamp,
        )

    def parse_query(self, query: str) -> ParsedQuery:
        raw = self._call_parser(query, kind="parse_query", ground_text=query)
        if raw is None:
            self.stats.n_failures += 1
            return ParsedQuery(evidence=[], task_signature="", source_text=query, parse_failed=True)
        units, _ = self._units(raw["evidence"], query)
        return ParsedQuery(evidence=units, task_signature=self._task(raw["task"]), source_text=query)

    # ----------------------------------------------------------- parsing
    def _call_parser(self, text: str, kind: str, ground_text: str) -> dict | None:
        rules = _MEMORY_RULES if kind == "parse_memory" else _QUERY_RULES
        messages = [
            {"role": "system", "content": _PARSE_SYSTEM.format(kind=kind, memory_rules=rules)},
            {"role": "user", "content": text},
        ]
        self.stats.n_parsed += 1
        for attempt in range(self.max_retries + 1):
            out = self.llm.chat(messages, temperature=0.0, json_mode=True, max_tokens=1500)
            try:
                obj = extract_json(out)
                self._validate(obj, kind)
                return obj
            except (ValueError, TypeError, KeyError) as exc:
                if attempt < self.max_retries:
                    self.stats.n_retries += 1
                    messages = messages + [
                        {"role": "assistant", "content": out},
                        {"role": "user", "content": f"Validation error: {exc}. Return corrected JSON only."},
                    ]
        return None

    @staticmethod
    def _validate(obj: dict, kind: str) -> None:
        if not isinstance(obj, dict):
            raise ValueError("top level must be an object")
        if not isinstance(obj.get("evidence"), list):
            raise ValueError("'evidence' must be a list")
        for i, e in enumerate(obj["evidence"]):
            for f in ("entity", "attribute", "value_type", "value", "span"):
                if not isinstance(e.get(f), (str, int, float)) or str(e.get(f)).strip() == "":
                    raise ValueError(f"evidence[{i}].{f} missing or empty")
            if str(e["value_type"]).lower() not in VALUE_TYPES:
                raise ValueError(f"evidence[{i}].value_type must be one of {VALUE_TYPES}")
        task = obj.get("task")
        if not isinstance(task, dict) or not task.get("target"):
            raise ValueError("'task' must contain 'operation' and 'target'")
        if kind == "parse_memory":
            sup = obj.get("support", [])
            if not isinstance(sup, list):
                raise ValueError("'support' must be a list of indices")
            n = len(obj["evidence"])
            if any(not isinstance(i, (int, float)) or not 0 <= int(i) < n for i in sup):
                raise ValueError("'support' contains an out-of-range index")

    def _units(self, records: list[dict], ground_text: str) -> tuple[list[EvidenceUnit], list[int]]:
        units: list[EvidenceUnit] = []
        kept: list[int] = []
        seen: set[EvidenceUnit] = set()
        g = _squash(ground_text)
        for i, rec in enumerate(records):
            span = str(rec["span"])
            if _squash(span) not in g:
                self.stats.n_ungrounded_units += 1
                continue
            vtype = str(rec["value_type"]).lower()
            key = self._canonical_key(str(rec["entity"]), str(rec["attribute"]), vtype)
            value = self._canonical_value(key, str(rec["value"]), vtype)
            unit = EvidenceUnit(key, value, span)
            if unit in seen:
                continue
            seen.add(unit)
            units.append(unit)
            kept.append(i)
        return units, kept

    def _outcome_evidence(self, rec, task: str) -> EvidenceUnit | None:
        if not isinstance(rec, dict):
            return None
        try:
            vtype = str(rec.get("value_type", "text")).lower()
            vtype = vtype if vtype in VALUE_TYPES else "text"
            entity = _snake(str(rec["entity"]))
            # Labelled by its original task so it is never confused with the
            # current task's answer (§3.3).
            key = KEY_SEP.join([entity, f"previous result: {task}", vtype])
            return EvidenceUnit(key, _normalize_value(str(rec["value"]), vtype))
        except KeyError:
            return None

    # ---------------------------------------------------- canonicalisation
    def _canonical_key(self, entity: str, attribute: str, vtype: str) -> str:
        raw = KEY_SEP.join([_snake(entity), _snake(attribute), vtype])
        return self._canonicalise(self.keys, raw, vtype, kind="key_equiv", what="evidence key")

    def _task(self, task: dict) -> str:
        raw = f"{_snake(str(task.get('operation', '')))}:{_snake(str(task['target']))}"
        return self._canonicalise(self.tasks, raw, "task", kind="task_equiv", what="task signature")

    def _canonicalise(self, reg: _Registry, raw: str, vtype: str, kind: str, what: str) -> str:
        if raw in reg.alias:
            return reg.alias[raw]
        cand = self._candidates(reg, raw, vtype)
        canonical = raw
        if cand:
            listing = "\n".join(f"[{i}] {reg.items[j]}" for i, j in enumerate(cand))
            prompt = _EQUIV_PROMPT.format(kind=kind, what=what, new=raw, existing=listing)
            try:
                out = extract_json(self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0, json_mode=True, max_tokens=50))
                idx = int(out.get("match", -1))
            except (ValueError, TypeError, AttributeError):
                idx = -1
            if 0 <= idx < len(cand):
                canonical = reg.items[cand[idx]]
                if kind == "key_equiv":
                    self.stats.n_key_merges += 1
                else:
                    self.stats.n_task_merges += 1
        if canonical == raw:
            reg.items.append(raw)
            reg.texts.append(raw.replace("_", " ").replace(KEY_SEP, " "))
            reg.types.append(vtype)
        reg.alias[raw] = canonical
        return canonical

    def _candidates(self, reg: _Registry, raw: str, vtype: str) -> list[int]:
        same_type = [i for i, t in enumerate(reg.types) if t == vtype]
        if not same_type:
            return []
        entity = raw.split(KEY_SEP)[0]
        exact_entity = [i for i in same_type if reg.items[i].split(KEY_SEP)[0] == entity]
        scored: list[tuple[float, int]] = []
        if self.encoder is not None:
            qv = self.encoder.encode([raw.replace("_", " ").replace(KEY_SEP, " ")])[0]
            mat = self.encoder.encode([reg.texts[i] for i in same_type])
            sims = mat @ qv
            scored = [(float(s), i) for s, i in zip(sims, same_type) if s >= self.candidate_sim]
        pool = {i: 1.0 for i in exact_entity}
        for s, i in scored:
            pool[i] = max(pool.get(i, 0.0), s)
        ranked = sorted(pool, key=lambda i: (-pool[i], i))
        return ranked[: self.n_candidates]

    def _canonical_value(self, key: str, value: str, vtype: str) -> str:
        v = _normalize_value(value, vtype)
        if vtype != "text":
            return v
        existing = self.values.setdefault(key, [])
        if v in existing or not existing:
            if v not in existing:
                existing.append(v)
            return v
        listing = "\n".join(f"[{i}] {x}" for i, x in enumerate(existing[-self.n_candidates :]))
        offset = max(0, len(existing) - self.n_candidates)
        prompt = _EQUIV_PROMPT.format(kind="value_equiv", what=f"value of '{key}'", new=v, existing=listing)
        try:
            out = extract_json(self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0, json_mode=True, max_tokens=50))
            idx = int(out.get("match", -1))
        except (ValueError, TypeError, AttributeError):
            idx = -1
        if 0 <= idx < min(self.n_candidates, len(existing)):
            return existing[offset + idx]
        existing.append(v)
        return v


# ------------------------------------------------------------- helpers
_BOOL = {"yes": "true", "true": "true", "1": "true", "y": "true", "valid": "true",
         "no": "false", "false": "false", "0": "false", "n": "false", "invalid": "false"}


def _snake(s: str) -> str:
    s = normalize(s)
    return re.sub(r"\s+", "_", s)


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _normalize_value(value: str, vtype: str) -> str:
    raw = str(value).strip()
    if vtype == "boolean":
        return _BOOL.get(normalize(raw), normalize(raw))
    if vtype == "numeric":
        s = raw.replace(",", "")
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if m:
            x = float(m.group())
            if "%" in s:
                x /= 100.0
            return f"{x:.4g}"
        return normalize(raw)
    if vtype == "temporal":
        for fmt in ("%Y-%m-%d", "%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%Y"):
            try:
                return datetime.strptime(raw, fmt).date().isoformat() if fmt != "%Y" else raw
            except ValueError:
                continue
        return normalize(raw)
    return normalize(raw)


def dump_record(rec: MemoryRecord) -> str:
    return json.dumps(
        {
            "id": rec.memory_id,
            "task": rec.task_signature,
            "evidence": [[u.key, u.value] for u in rec.evidence],
            "support": [[u.key, u.value] for u in rec.support_evidence],
            "outcome": rec.outcome,
            "outcome_evidence": None if rec.outcome_evidence is None else [rec.outcome_evidence.key, rec.outcome_evidence.value],
            "parse_failed": rec.parse_failed,
        },
        ensure_ascii=False,
    )
