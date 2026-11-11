import json
import re

from . import prompts
from .schema import EvidenceUnit, ParsedRecord, TaskSignature

_NUMERIC_RE = re.compile(r"-?\d+(\.\d+)?")


def normalize_value(value_type, raw_value):
    text = raw_value.strip()
    if value_type == "boolean":
        low = text.lower()
        if low in ("true", "yes", "valid"):
            return "true"
        if low in ("false", "no", "invalid"):
            return "false"
        return low
    if value_type == "numeric":
        is_percent = "%" in text
        match = _NUMERIC_RE.search(text.replace(",", ""))
        if match:
            num = float(match.group(0))
            return f"{num}%" if is_percent else repr(num)
        return text.lower()
    if value_type == "temporal":
        digits = re.sub(r"[^0-9]", "", text)
        return digits if digits else text.lower()
    if value_type == "categorical":
        return text.lower().strip()
    return text.lower().strip()


def _find_span(source_text, span):
    return bool(span) and span.lower() in source_text.lower()


def _well_formed_unit(unit, source_text):
    return (
        _find_span(source_text, unit.get("span", ""))
        and unit.get("value", "").strip() != ""
        and unit.get("entity", "").strip() != ""
        and unit.get("attribute", "").strip() != ""
    )


def drop_malformed_evidence(raw, source_text):
    kept_evidence = [u for u in raw.get("evidence", []) if _well_formed_unit(u, source_text)]
    kept_spans = {u["span"] for u in kept_evidence}
    cleaned = dict(raw)
    cleaned["evidence"] = kept_evidence
    cleaned["support_spans"] = [s for s in raw.get("support_spans", []) if s in kept_spans]
    return cleaned


def structural_errors(raw):
    errors = []
    if not raw.get("task_operation", "").strip() or not raw.get("task_target", "").strip():
        errors.append("task_operation/task_target must not be empty")
    if raw.get("has_outcome") and not raw.get("outcome_text", "").strip():
        errors.append("has_outcome is true but outcome_text is empty")
    if raw.get("outcome_statable_as_fact") and not raw.get("outcome_evidence_entity", "").strip():
        errors.append("outcome_statable_as_fact is true but outcome_evidence_entity is empty")
    return errors


def _build_record(raw, source_text):
    evidence = []
    for unit in raw.get("evidence", []):
        evidence.append(
            EvidenceUnit(
                entity=unit["entity"].strip(),
                attribute=unit["attribute"].strip(),
                value_type=unit["value_type"],
                value=unit["value"].strip(),
                span=unit["span"],
                multi_valued=bool(unit["multi_valued"]),
            )
        )
    support_spans = set(raw.get("support_spans", []))
    support_mask = [u.span in support_spans for u in evidence]

    task = TaskSignature(
        operation=raw.get("task_operation", "").strip().lower(),
        target=raw.get("task_target", "").strip().lower(),
    )

    outcome_text = raw.get("outcome_text", "").strip()
    outcome_text = outcome_text if raw.get("has_outcome") and outcome_text else None

    outcome_evidence = None
    if raw.get("outcome_statable_as_fact") and raw.get("outcome_evidence_entity", "").strip():
        outcome_evidence = EvidenceUnit(
            entity=raw["outcome_evidence_entity"].strip(),
            attribute=raw["outcome_evidence_attribute"].strip(),
            value_type=raw.get("outcome_evidence_value_type", "categorical") or "categorical",
            value=raw["outcome_evidence_value"].strip(),
            span="",
            multi_valued=False,
        )

    return ParsedRecord(
        source_text=source_text,
        evidence=evidence,
        task=task,
        support_mask=support_mask,
        outcome_text=outcome_text,
        outcome_evidence=outcome_evidence,
        parse_ok=True,
    )


def _one_attempt(client, text, retry_error=None):
    messages = prompts.build_parse_messages(text, retry_error=retry_error)
    response = client.chat(messages, response_format=prompts.PARSE_SCHEMA, max_tokens=1536, temperature=0.0, seed=0)
    try:
        raw = json.loads(response["content"])
    except (json.JSONDecodeError, TypeError):
        return None, ["response was not valid JSON"]
    cleaned = drop_malformed_evidence(raw, text)
    errors = structural_errors(cleaned)
    return cleaned, errors


def parse_text(client, text):
    cleaned, errors = _one_attempt(client, text)
    if errors:
        cleaned, errors = _one_attempt(client, text, retry_error="; ".join(errors))
    if cleaned is None or errors:
        return ParsedRecord(source_text=text, evidence=[], task=None, support_mask=[], outcome_text=None, outcome_evidence=None, parse_ok=False)
    return _build_record(cleaned, text)
