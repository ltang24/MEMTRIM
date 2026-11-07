

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .core import MemTrim, render_context
from .embed import Encoder
from .llm import LLM
from .memory.base import Retrieved
from .parser import EvidenceParser
from .schema import MemTrimOptions, ParsedQuery
from .text import normalize, parse_yes_no

SYSTEM = "You are a helpful assistant that answers the user's current question. <<MOCK:answer>>"


def answer_instruction(answer_type: str) -> str:
    if answer_type == "yesno":
        return "Answer with yes or no only."
    return "Give only the final answer (an entity name), with no explanation."


@dataclass
class QueryCtx:
    item: dict
    retrieved: list[Retrieved]  # ranked, possibly longer than k (MMR pool)
    k: int
    lookup: dict[str, tuple[str, str]]  # interaction_id -> (x_m, y_m)
    llm: LLM
    encoder: Encoder
    memtrim: MemTrim | None = None  # per-store instance holding parsed memories
    parser: EvidenceParser | None = None
    parsed_query: ParsedQuery | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def top(self) -> list[Retrieved]:
        return self.retrieved[: self.k]

    @property
    def atype(self) -> str:
        return self.item["answer_type"]


# ------------------------------------------------------------------ helpers
def _ask(ctx: QueryCtx, memory_block: str | None, extra: str = "", query: str | None = None, system: str = SYSTEM) -> str:
    query = query if query is not None else ctx.item["query"]
    parts = []
    if memory_block:
        parts.append(f"Relevant memories from past interactions:\n{memory_block}\n")
    parts.append(f"Current query:\n{query}\n")
    if extra:
        parts.append(extra)
    parts.append(answer_instruction(ctx.atype))
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(parts)}]
    return clean_answer(ctx.llm.chat(msgs, max_tokens=ctx.params.get("answer_max_tokens", 64)))


def clean_answer(text: str) -> str:
    text = re.sub(r"(?s)<think>.*?</think>", "", text).strip()
    m = re.search(r"(?i)final answer\s*[:\-]\s*(.+)", text)
    if m:
        return m.group(1).strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def _block(texts: list[str]) -> str:
    return "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts) if t.strip())


def _strip_answer(text: str) -> str:
    return re.sub(r"(?im)^\s*answer\s*:.*$", "", text).strip()


# ----------------------------------------------------------- §2 conditions
def no_memory(ctx: QueryCtx) -> str:
    return _ask(ctx, None)


def plain_memory(ctx: QueryCtx) -> str:
    return _ask(ctx, _block([r.text for r in ctx.top]))


# ----------------------------------------------------------------- baselines
def facts_only(ctx: QueryCtx) -> str:
    """Remove the stored answer, keep only factual content: each retrieved
    item is replaced by its source interaction x_m without y_m."""
    texts = []
    for r in ctx.top:
        if r.interaction_id in ctx.lookup:
            texts.append(_strip_answer(ctx.lookup[r.interaction_id][0]))
        else:
            texts.append(_strip_answer(r.text))
    return _ask(ctx, _block(texts))


def top1(ctx: QueryCtx) -> str:
    return _ask(ctx, _block([r.text for r in ctx.retrieved[:1]]))


def mmr(ctx: QueryCtx) -> str:
    """Carbonell & Goldstein (1998): rerank a larger pool, lambda trades off
    relevance and redundancy."""
    lam = ctx.params.get("mmr_lambda", 0.5)
    pool = ctx.retrieved
    if not pool:
        return _ask(ctx, None)
    E = ctx.encoder.encode([r.text for r in pool])
    q = ctx.encoder.encode([ctx.item["query"]])[0]
    rel = E @ q
    chosen: list[int] = []
    cand = list(range(len(pool)))
    while cand and len(chosen) < ctx.k:
        def score(i: int) -> float:
            red = max((float(E[i] @ E[j]) for j in chosen), default=0.0)
            return lam * float(rel[i]) - (1 - lam) * red
        best = max(cand, key=lambda i: (score(i), -i))
        chosen.append(best)
        cand.remove(best)
    return _ask(ctx, _block([pool[i].text for i in chosen]))


def sim_filter(ctx: QueryCtx) -> str:
    tau = ctx.params.get("sim_threshold", 0.5)
    if not ctx.top:
        return _ask(ctx, None)
    sims = ctx.encoder.encode([r.text for r in ctx.top]) @ ctx.encoder.encode([ctx.item["query"]])[0]
    kept = [r.text for r, s in zip(ctx.top, sims) if s >= tau]
    return _ask(ctx, _block(kept) if kept else None)


_RECOMP_MODEL: dict[str, Any] = {}


def _recomp_scores(name: str, query: str, sents: list[str], encoder: Encoder) -> np.ndarray:
    if name in ("", None, "encoder"):
        return encoder.encode(sents) @ encoder.encode([query])[0]
    import torch
    from transformers import AutoModel, AutoTokenizer

    if name not in _RECOMP_MODEL:
        _RECOMP_MODEL[name] = (AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval())
    tok, model = _RECOMP_MODEL[name]

    def emb(texts):
        b = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        return ((h * m).sum(1) / m.sum(1)).numpy()  # mean pooling (Contriever)

    return emb(sents) @ emb([query])[0]


def recomp(ctx: QueryCtx) -> str:
    """RECOMP extractive compressor (Xu et al., 2024): keep the top-n sentences
    of the retrieved context by query similarity under the released
    Contriever-based extractive compressor."""
    sents = [s.strip() for r in ctx.top for s in re.split(r"(?<=[.?!])\s+|\n+", r.text) if s.strip()]
    if not sents:
        return _ask(ctx, None)
    scores = _recomp_scores(ctx.params.get("recomp_model", "fangyuan/nq_extractive_compressor"), ctx.item["query"], sents, ctx.encoder)
    n = ctx.params.get("recomp_top_sentences", 3)
    keep = sorted(np.argsort(-scores, kind="stable")[:n])
    return _ask(ctx, "\n".join(sents[i] for i in keep))


_LINGUA: dict[str, Any] = {}


def llmlingua2(ctx: QueryCtx) -> str:
    """LLMLingua-2 (Pan et al., 2024) token-level compression of the retrieved
    context at a fixed rate."""
    if not ctx.top:
        return _ask(ctx, None)
    name = ctx.params.get("llmlingua_model", "microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
    if name not in _LINGUA:
        from llmlingua import PromptCompressor

        _LINGUA[name] = PromptCompressor(model_name=name, use_llmlingua2=True, device_map=ctx.params.get("device", "cpu"))
    out = _LINGUA[name].compress_prompt(
        _block([r.text for r in ctx.top]), rate=ctx.params.get("llmlingua_rate", 0.5), force_tokens=["\n", "?"]
    )
    return _ask(ctx, out["compressed_prompt"])


def context_faithful(ctx: QueryCtx) -> str:
    """Context-faithful prompting (Zhou et al., 2023), adapted so the current
    query is the context to be faithful to and memories are secondary."""
    sys = (SYSTEM + " The information stated in the CURRENT QUERY is the ground truth for this question. "
           "Retrieved memories describe earlier interactions; whenever a memory disagrees with the current query, "
           "or was about a different question, follow the current query and ignore the memory.")
    return _ask(ctx, _block([r.text for r in ctx.top]),
                extra="Based only on the current query (use memories only if they are consistent with it), what is the answer?",
                system=sys)


def answer_first(ctx: QueryCtx) -> str:
    """Answer from the current query first, then compare with retrieved
    information and revise only if it is consistent (Neeman et al., 2023)."""
    first = _ask(ctx, None)
    extra = (f"An answer derived from the current query alone is: {first}\n"
             "Compare it with the retrieved memories. Change it ONLY if a memory adds information that is "
             "consistent with the current query and shows the first answer is wrong; otherwise keep it.")
    return _ask(ctx, _block([r.text for r in ctx.top]), extra=extra)


def micro_act(ctx: QueryCtx) -> str:
    """Micro-Act (Huo et al., 2025) style: an explicit, decomposed reasoning
    step that lists conflicts between retrieved knowledge and the current
    input before answering."""
    block = _block([r.text for r in ctx.top])
    prompt = (
        "<<MOCK:judge_conflict>>Compare the retrieved memories with the current query step by step.\n"
        "1. List each fact or question in the memories.\n2. For each, say whether it is the SAME as, "
        "CONFLICTS with, or is ABSENT from the current query, and whether the memory answered the same question.\n"
        "3. State which memory content can safely be used.\n\n"
        f"Retrieved memories:\n{block}\n\nCurrent query:\n{ctx.item['query']}"
    )
    analysis = ctx.llm.chat([{"role": "user", "content": prompt}], max_tokens=ctx.params.get("analysis_max_tokens", 512))
    return _ask(ctx, block, extra=f"Conflict analysis:\n{analysis}\n")


# ------------------------------------------------------------------ MEMTRIM
def memtrim(ctx: QueryCtx, options: MemTrimOptions | None = None) -> str:
    assert ctx.memtrim is not None and ctx.parsed_query is not None
    mt = ctx.memtrim
    if options is not None:
        mt = MemTrim(options, encode=ctx.encoder.encode)
        mt.memories, mt.trie = ctx.memtrim.memories, ctx.memtrim.trie
    ids = [r.interaction_id for r in ctx.top if r.interaction_id in mt.memories]
    ids = list(dict.fromkeys(ids))
    context = mt.build_context(ctx.parsed_query, ids)
    block = render_context(context)
    return _ask(ctx, block or None)


def memtrim_ablation(name: str) -> Callable[[QueryCtx], str]:
    opts = MemTrimOptions.ablation(name)
    return lambda ctx: memtrim(ctx, opts)


# ----------------------------------------------------------------- defenses
def robustrag(ctx: QueryCtx) -> str:
    """RobustRAG (Xiang et al., 2024) isolate-then-aggregate: answer with each
    retrieved item alone, then aggregate. Yes/no -> majority vote; entities ->
    keyword aggregation with threshold alpha; fall back to no-memory."""
    answers = [_ask(ctx, _block([r.text])) for r in ctx.top]
    if not answers:
        return _ask(ctx, None)
    alpha = ctx.params.get("robustrag_alpha", 0.3)
    if ctx.atype == "yesno":
        votes = collections.Counter(parse_yes_no(a) for a in answers if parse_yes_no(a))
        return votes.most_common(1)[0][0] if votes else _ask(ctx, None)
    counts = collections.Counter(normalize(a) for a in answers if normalize(a))
    if counts:
        best, n = counts.most_common(1)[0]
        if n >= alpha * len(answers):
            return best
    return _ask(ctx, None)


def trustrag(ctx: QueryCtx) -> str:
    """TrustRAG (Zhou et al., 2025): (1) K-means (k=2) over retrieved
    embeddings; a cluster whose items are near-duplicates (mean pairwise cosine
    >= tau) is treated as injected and removed; (2) internal-knowledge
    consolidation: the model's own answer is compared with the remaining
    memories before the final answer."""
    items = list(ctx.top)
    tau = ctx.params.get("trustrag_tau", 0.88)
    E = ctx.encoder.encode([r.text for r in items]) if items else None
    if len(items) >= 3 and len(np.unique(np.round(E, 6), axis=0)) >= 2:
        from sklearn.cluster import KMeans

        lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(E)
        keep = []
        for c in (0, 1):
            idx = [i for i in range(len(items)) if lab[i] == c]
            if len(idx) >= 2:
                sub = E[idx] @ E[idx].T
                mean = (sub.sum() - len(idx)) / (len(idx) * (len(idx) - 1))
                if mean >= tau:
                    continue
            keep += idx
        items = [items[i] for i in sorted(keep)]
    internal = _ask(ctx, None)
    extra = (f"Your own answer from internal knowledge and the query alone: {internal}\n"
             "Consolidate: identify memory content consistent with the current query, discard conflicting or "
             "unreliable content, then give the final answer.")
    return _ask(ctx, _block([r.text for r in items]) if items else None, extra=extra)


def amemguard(ctx: QueryCtx) -> str:
    """A-MemGuard (Wei et al., 2025) consensus-based validation: derive an
    answer from each retrieved memory, flag memories whose induced answer
    deviates from the consensus, and answer with the remaining memories."""
    items = list(ctx.top)
    if not items:
        return _ask(ctx, None)
    induced = [normalize(_ask(ctx, _block([r.text]))) for r in items]
    if ctx.atype == "yesno":
        induced = [parse_yes_no(a) or a for a in induced]
    base = normalize(_ask(ctx, None))
    if ctx.atype == "yesno":
        base = parse_yes_no(base) or base
    counts = collections.Counter(induced + [base])  # query-only path joins the vote
    consensus = counts.most_common(1)[0][0]
    kept = [r.text for r, a in zip(items, induced) if a == consensus]
    return _ask(ctx, _block(kept) if kept else None)


METHODS: dict[str, Callable[[QueryCtx], str]] = {
    "no_memory": no_memory,
    "memory": plain_memory,
    "facts_only": facts_only,
    "top1": top1,
    "mmr": mmr,
    "sim_filter": sim_filter,
    "recomp": recomp,
    "llmlingua2": llmlingua2,
    "context_faithful": context_faithful,
    "answer_first": answer_first,
    "micro_act": micro_act,
    "memtrim": memtrim,
    "memtrim-keep_shared": memtrim_ablation("keep_shared"),
    "memtrim-keep_conflicts": memtrim_ablation("keep_conflicts"),
    "memtrim-keep_old_answer": memtrim_ablation("keep_old_answer"),
    "memtrim-ignore_task_change": memtrim_ablation("ignore_task_change"),
    "memtrim-embedding_match": memtrim_ablation("embedding_match"),
    "robustrag": robustrag,
    "trustrag": trustrag,
    "amemguard": amemguard,
}

NEEDS_MEMTRIM = {m for m in METHODS if m.startswith("memtrim")}
