

from __future__ import annotations

import re
import string
import unicodedata

# A compact English stop-word list (function words only), used to define
# "content words" for the query-memory Jaccard overlap in RQ2.
STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers herself him himself his how
    i if in into is it its itself just me more most my myself no nor not now of off on
    once only or other our ours ourselves out over own same she should so some such than
    that the their theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves question answer context premises premise
    conclusion""".split()
)

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_WS = re.compile(r"\s+")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _PUNCT.sub(" ", text)
    text = _ARTICLES.sub(" ", text)
    return _WS.sub(" ", text).strip()


def content_words(text: str) -> set[str]:
    return {t for t in normalize(text).split() if t not in STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def query_memory_overlap(query_text: str, memory_text: str, stored_answer: str | None) -> float:
    """Content-word Jaccard between query and memory, excluding the stored answer."""
    mem = memory_text
    if stored_answer:
        mem = re.sub(r"(?im)^\s*answer\s*:.*$", " ", mem)
    q = content_words(query_text)
    m = content_words(mem)
    if stored_answer and normalize(stored_answer) not in {"yes", "no"}:
        m -= content_words(stored_answer) - q
    return jaccard(q, m)


# ----------------------------------------------------------------- scoring
_YES = re.compile(r"\b(yes|true|valid|correct)\b")
_NO = re.compile(r"\b(no|false|invalid|incorrect|not)\b")


def parse_yes_no(prediction: str) -> str | None:
    p = normalize(prediction)
    head = p.split("\n")[0]
    m_final = re.search(r"final answer\s*(?:is)?\s*(yes|no)", p)
    if m_final:
        return m_final.group(1)
    first = re.match(r"\s*(yes|no)\b", head)
    if first:
        return first.group(1)
    y, n = _YES.search(head), _NO.search(head)
    if y and not n:
        return "yes"
    if n and not y:
        return "no"
    if y and n:
        return "yes" if y.start() < n.start() else "no"
    return None


def is_correct(prediction: str, gold: str, aliases: list[str] | None, answer_type: str) -> bool:
    if answer_type == "yesno":
        return parse_yes_no(prediction) == normalize(gold)
    pred = normalize(prediction)
    if not pred:
        return False
    for g in [gold, *(aliases or [])]:
        g = normalize(g)
        if g and (g in pred or (len(pred) >= 3 and pred in g and len(pred) / max(len(g), 1) > 0.5)):
            return True
    return False
