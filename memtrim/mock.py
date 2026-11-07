"""Offline mock LLM responses for unit tests and ``--smoke`` runs.

This is a heuristic text processor, not a model. It exists so the whole
pipeline (data -> memory -> methods -> metrics -> tables/figures) can be
exercised without GPUs or API keys. Numbers produced with it are meaningless.
"""

from __future__ import annotations

import json
import re

_SENT = re.compile(r"(?<=[.?!])\s+|\n+")
_COPULA = re.compile(r"^(.*?)\s+\b(is|are|was|were|follows|plays for)\b\s+(.*)$", re.I)


def _evidence(text: str) -> list[dict]:
    body = re.split(r"\n\s*ANSWER:", text)[0]
    out = []
    for sent in _SENT.split(body):
        s = sent.strip().rstrip(".?!")
        if not s or s.lower().startswith(("question", "q:", "answer", "complete")):
            continue
        s = re.sub(r"^(context|premises?|setup|conclusion)\s*:\s*", "", s, flags=re.I)
        m = _COPULA.match(s)
        if not m:
            continue
        left, rel, right = m.group(1), m.group(2), m.group(3)
        out.append({"entity": left[-60:], "attribute": rel, "value_type": "categorical",
                    "value": right[:80], "span": s})
    return out


def _task(text: str) -> dict:
    q = ""
    for line in text.splitlines():
        if re.match(r"\s*(question|q)\s*:", line, re.I):
            q = line.split(":", 1)[1]
    q = re.sub(r"\b[A-Z0-9][\w+\-]*\b", "", q)  # drop names / numbers
    q = re.sub(r"[^a-z ]", " ", q.lower())
    target = "_".join(q.split()[:6]) or "unknown"
    return {"operation": "answer", "target": target}


def mock_dispatch(kind: str, text: str) -> str:
    if kind in ("parse_memory", "parse_query"):
        ev = _evidence(text)
        obj: dict = {"evidence": ev, "task": _task(text)}
        if kind == "parse_memory":
            obj["support"] = list(range(len(ev)))
            ans = re.search(r"ANSWER:\s*(.+)", text)
            obj["answer_as_evidence"] = (
                {"entity": "answer", "attribute": "value", "value_type": "text", "value": ans.group(1).strip()}
                if ans else None
            )
        return json.dumps(obj)
    if kind in ("key_equiv", "task_equiv", "value_equiv"):
        return json.dumps({"match": -1})
    if kind == "judge_conflict":
        return "No conflict."
    # Answering: echo a stored answer if one is visible (to mimic memory reuse),
    # otherwise a fixed guess.
    if re.search(r"answer (yes or no|with yes)", text, re.I):
        m = re.findall(r"(?:Answer|answer to this same question)\s*:\s*(yes|no)\b", text, re.I)
        return (m[-1] if m else "no").capitalize()
    m = re.findall(r"Answer\s*:\s*([^\n]+)", text)
    return m[-1].strip() if m else "unknown"
