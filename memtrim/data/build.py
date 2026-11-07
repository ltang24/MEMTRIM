from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import re
import zipfile
from pathlib import Path

import numpy as np

from ..embed import get_encoder
from ..text import content_words, jaccard, normalize

SEED = 42
MAX_PER_CONDITION = 500


# =========================================================== RippleEdits (C.1)
def _target_from_prompt(prompt: str, other_prompt: str) -> str:
    """The edit and original prompts share a template and differ in the
    target; recover the target as the differing suffix."""
    p, o = prompt.rstrip(". "), other_prompt.rstrip(". ")
    i = 0
    while i < min(len(p), len(o)) and p[i] == o[i]:
        i += 1
    # back up to a word boundary
    while i > 0 and p[i - 1] not in " ":
        i -= 1
    return p[i:].strip()


def _cloze_question(prompt: str) -> str:
    return f'Complete the statement with the correct entity: "{prompt.strip()} ___"'


def build_rippleedits(raw_dir: Path, n_context_facts: int = 3) -> list[dict]:
    root = raw_dir / "RippleEdits" / "data" / "benchmark"
    edits = []
    for subset in ("random", "popular"):  # RECENT has no original value -> excluded
        for ex in json.loads((root / f"{subset}.json").read_text()):
            ex["_subset"] = subset
            edits.append(ex)

    items: list[dict] = []
    for idx, ex in enumerate(edits):
        edit = ex["edit"]
        orig = edit.get("original_fact")
        if not orig:
            continue
        new_t = _target_from_prompt(edit["prompt"], orig["prompt"])
        old_t = _target_from_prompt(orig["prompt"], edit["prompt"])
        alias_tests = [q for t in ex.get("Subject_Aliasing", []) for q in t["test_queries"]]
        if not alias_tests or not new_t or not old_t:
            continue
        q0 = alias_tests[0]
        new_aliases = [a for ans in q0["answers"] for a in [ans["value"], *ans.get("aliases", [])]]
        if normalize(new_t) == normalize(old_t) or normalize(old_t) in {normalize(a) for a in new_aliases}:
            continue  # indistinguishable after alias normalisation
        # Other facts about the same subject (unchanged by the edit) give the
        # stored interaction memory-only context.
        ctx = []
        for t in ex.get("Relation_Specificity", []) + ex.get("Forgetfulness", []):
            for q in t["test_queries"]:
                if q.get("answers"):
                    ctx.append(f"{q['prompt'].strip()} {q['answers'][0]['value']}.")
        ctx = sorted(set(ctx))[:n_context_facts]
        items.append(dict(
            src=idx, subset=ex["_subset"], relation=edit["relation"],
            edit_fact=edit["prompt"].strip(), orig_fact=orig["prompt"].strip(),
            new_t=new_t, old_t=old_t, new_aliases=new_aliases, ctx=ctx,
            question=_cloze_question(q0["prompt"]),
        ))

    rng = random.Random(SEED)
    rng.shuffle(items)
    items = items[:MAX_PER_CONDITION]
    items.sort(key=lambda x: x["src"])

    out = []
    for it in items:
        pid = f"re-{it['src']:05d}"
        query = f"Context: {it['edit_fact']}\nQuestion: {it['question']}"
        for cond in ("TA", "TD"):
            fact = it["edit_fact"] if cond == "TA" else it["orig_fact"]
            answer = it["new_t"] if cond == "TA" else it["old_t"]
            mem_text = "Context: " + " ".join([fact, *it["ctx"]]) + f"\nQuestion: {it['question']}"
            out.append(dict(
                id=f"{pid}-{cond}", pair_id=pid, dataset="rippleedits", condition=cond,
                group=f"rel:{it['relation']}", query=query, question=it["question"],
                gold=it["new_t"], aliases=it["new_aliases"], answer_type="entity",
                memory=dict(id=f"{pid}-{cond}-m", text=mem_text, answer=answer),
                controlled=dict(
                    units=[dict(mem_text=it["orig_fact"], td_text=it["edit_fact"], support=True)]
                    + [dict(mem_text=c, td_text=None, support=False) for c in it["ctx"]],
                    question_ta=it["question"], question_td=it["question"],
                    ta_gold=it["old_t"], ta_aliases=[it["old_t"]],
                    td_gold=it["new_t"], td_aliases=it["new_aliases"],
                    answer_type="entity",
                ),
            ))
    return out


# ========================================================== CounterLogic (C.2)
VALIDITY_Q = "Is the argument logically valid, i.e., does the conclusion follow from the premises? Answer yes or no."
KNOWLEDGE_Q = "Does the conclusion agree with real-world knowledge? Answer yes or no."


def _yn(b: bool) -> str:
    return "yes" if b else "no"


def _cluster(emb: np.ndarray, threshold: float, min_size: int) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=threshold
    ).fit_predict(emb)
    counts = collections.Counter(labels)
    big = [c for c, n in counts.items() if n >= min_size]
    if not big:
        return np.zeros(len(labels), dtype=int)
    cents = {c: emb[labels == c].mean(0) for c in big}
    cents = {c: v / (np.linalg.norm(v) + 1e-12) for c, v in cents.items()}
    out = labels.copy()
    for i, c in enumerate(labels):
        if counts[c] < min_size:  # attach small clusters to nearest large one
            out[i] = max(big, key=lambda b: float(emb[i] @ cents[b]))
    return out


def build_counterlogic(raw_dir: Path, encoder: str) -> list[dict]:
    path = raw_dir / "CounterLogic" / "Dataset" / "data_v3.json"
    data = json.loads(path.read_text())
    insts = []
    for schema, lst in sorted(data.items(), key=lambda kv: int(kv[0])):
        for j, x in enumerate(lst):
            insts.append(dict(src=f"{schema}-{j}", schema=schema, **x))
    texts = [" ".join(x["premise_list"]) + " " + x["conclusion"] for x in insts]  # no task/label
    emb = get_encoder(encoder).encode(texts)
    labels = _cluster(emb, threshold=0.20, min_size=5)

    ta = [i for i, x in enumerate(insts) if x["validity"] == x["believability"]]
    td = [i for i, x in enumerate(insts) if x["validity"] != x["believability"]]
    rng = random.Random(SEED)
    rng.shuffle(ta)
    rng.shuffle(td)
    chosen = sorted(ta[:MAX_PER_CONDITION]) + sorted(td[:MAX_PER_CONDITION])

    out = []
    for i in chosen:
        x = insts[i]
        cond = "TA" if x["validity"] == x["believability"] else "TD"
        prem = " ".join(p.rstrip(".") + "." for p in x["premise_list"])
        body = f"Premises: {prem}\nConclusion: {x['conclusion'].rstrip('.')}."
        pid = f"cl-{x['src']}"
        out.append(dict(
            id=f"{pid}-{cond}", pair_id=pid, dataset="counterlogic", condition=cond,
            group=f"cluster:{int(labels[i])}", query=f"{body}\nQuestion: {KNOWLEDGE_Q}",
            question=KNOWLEDGE_Q, gold=_yn(x["believability"]), aliases=[], answer_type="yesno",
            memory=dict(id=f"{pid}-{cond}-m", text=f"{body}\nQuestion: {VALIDITY_Q}", answer=_yn(x["validity"])),
            controlled=dict(
                units=[dict(mem_text=p.rstrip(".") + ".", td_text=None, support=False) for p in x["premise_list"]]
                + [dict(mem_text=f"Conclusion: {x['conclusion'].rstrip('.')}.", td_text=None, support=False, always=True),
                   dict(mem_text="", td_text="", support=True, is_task=True)],
                question_ta=VALIDITY_Q, question_td=KNOWLEDGE_Q,
                ta_gold=_yn(x["validity"]), ta_aliases=[], td_gold=_yn(x["believability"]), td_aliases=[],
                answer_type="yesno",
            ),
        ))
    return out


# ============================================================ CLADDER (C.3)
_NUM = re.compile(r"\d+(?:\.\d+)?%?")


def _flatten(d, prefix=""):
    if isinstance(d, dict):
        for k in d:
            yield from _flatten(d[k], f"{prefix}{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        try:
            yield prefix, round(float(d), 2)
        except (TypeError, ValueError):  # e.g. backadj lists variable names
            yield prefix, str(d)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=\.)\s+", text) if s.strip()]


def build_cladder(raw_dir: Path, encoder: str, release: str = "balanced", same_question: bool = False) -> list[dict]:
    zpath = raw_dir / "cladder" / "data" / "cladder-v1.zip"
    with zipfile.ZipFile(zpath) as z:
        qs = json.loads(z.read(f"cladder-v1-q-{release}.json"))
        metas = {m["model_id"]: m for m in json.loads(z.read("cladder-v1-meta-models.json"))}
    for x in qs:
        x["_bg"] = metas[x["meta"]["model_id"]]["background"]
        x["_ev"] = dict(_flatten(x["meta"].get("given_info", {})))
        x["_reason"] = " ".join(str((x["reasoning"] or {}).get(k, "")) for k in ("step4", "step5", "end"))

    groups = collections.defaultdict(list)
    for x in qs:
        m = x["meta"]
        groups[(m["story_id"], m["graph_id"], m["query_type"], m["rung"])].append(x)

    enc = get_encoder(encoder)
    sources = []
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda x: x["question_id"])
        if len(g) < 3:
            continue
        masked = [_NUM.sub("<NUM>", f"{x['_bg']} {x['given_info']} {x['question']}") for x in g]
        emb = enc.encode(masked)
        sim = emb @ emb.T
        for a, s in enumerate(g):
            cands = {"TA": [], "TD": []}
            for b, c in enumerate(g):
                if a == b or sim[a, b] < 0.85:
                    continue
                # Table 5: only the causal values change; the question is fixed.
                if same_question and normalize(c["question"]) != normalize(s["question"]):
                    continue
                ks, kc = set(s["_ev"]), set(c["_ev"])
                jac = len(ks & kc) / max(len(ks | kc), 1)
                if jac < 0.80:
                    continue
                common = ks & kc
                unchanged = [k for k in common if s["_ev"][k] == c["_ev"][k]]
                changed = [k for k in common if s["_ev"][k] != c["_ev"][k]]
                if not unchanged or not changed:
                    continue
                if not any(
                    isinstance(c["_ev"][k], float) and f"{c['_ev'][k]:.2f}" in c["_reason"] for k in changed
                ):
                    continue
                cond = "TA" if s["answer"] == c["answer"] else "TD"
                cands[cond].append((float(sim[a, b]), jac, c["question_id"], c))
            if not cands["TA"] or not cands["TD"]:
                continue
            best = min(
                itertools.product(cands["TA"], cands["TD"]),
                key=lambda p: (abs(p[0][0] - p[1][0]) + abs(p[0][1] - p[1][1]), p[0][2], p[1][2]),
            )
            sources.append((s, best[0][3], best[1][3]))

    rng = random.Random(SEED)
    rng.shuffle(sources)
    sources = sorted(sources[:MAX_PER_CONDITION], key=lambda t: t[0]["question_id"])

    def text(x):
        return f"Context: {x['_bg']} {x['given_info']}\nQuestion: {x['question']}"

    out = []
    for s, ta_c, td_c in sources:
        pid = f"cd-{s['question_id']}"
        for cond, c in (("TA", ta_c), ("TD", td_c)):
            item = dict(
                id=f"{pid}-{cond}", pair_id=pid, dataset="cladder", condition=cond,
                group=f"story:{s['meta']['story_id']}", query=text(c), question=c["question"],
                gold=c["answer"], aliases=[], answer_type="yesno",
                memory=dict(id=f"{pid}-{cond}-m", text=text(s), answer=s["answer"]),
                controlled=None,
            )
            if cond == "TD":
                ss, cs = _sentences(s["given_info"]), _sentences(c["given_info"])
                if len(ss) == len(cs):
                    units = [dict(mem_text=s["_bg"], td_text=None, support=False)]
                    for a, b in zip(ss, cs):
                        same = _NUM.findall(a) == _NUM.findall(b)
                        units.append(dict(mem_text=a, td_text=None if same else b, support=not same))
                    item["controlled"] = dict(
                        units=units, question_ta=s["question"], question_td=c["question"],
                        ta_gold=s["answer"], ta_aliases=[], td_gold=c["answer"], td_aliases=[],
                        answer_type="yesno",
                    )
            out.append(item)
    return out


# ======================================================= checks (C.4)
def balance_report(items: list[dict]) -> dict:
    """Compare query-memory similarity, overlap and length between TA and TD."""
    stats = collections.defaultdict(lambda: collections.defaultdict(list))
    for it in items:
        q, m = it["query"], it["memory"]["text"]
        stats[it["condition"]]["overlap"].append(jaccard(content_words(q), content_words(m)))
        stats[it["condition"]]["len_query"].append(len(q.split()))
        stats[it["condition"]]["len_memory"].append(len(m.split()))
    rep = {}
    for metric in ("overlap", "len_query", "len_memory"):
        a, b = np.array(stats["TA"][metric]), np.array(stats["TD"][metric])
        if len(a) and len(b):
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12
            rep[metric] = dict(TA=float(a.mean()), TD=float(b.mean()), smd=float((a.mean() - b.mean()) / pooled))
    return rep


def dedupe(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for it in items:
        k = (it["condition"], it["query"], it["memory"]["text"])
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all", "rippleedits", "counterlogic", "cladder"])
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--cladder-release", default="balanced")
    ap.add_argument("--cladder-same-question", action="store_true",
                    help="Also require identical question text (only causal values differ). "
                         "Not stated in App. C.3; yields fewer pairs.")
    args = ap.parse_args()
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    todo = ["rippleedits", "counterlogic", "cladder"] if args.dataset == "all" else [args.dataset]
    for name in todo:
        if name == "rippleedits":
            items = build_rippleedits(raw)
        elif name == "counterlogic":
            items = build_counterlogic(raw, args.encoder)
        else:
            items = build_cladder(raw, args.encoder, args.cladder_release, args.cladder_same_question)
        items = dedupe(items)
        path = out / f"{name}.jsonl"
        with path.open("w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        n = collections.Counter(it["condition"] for it in items)
        print(f"[{name}] wrote {len(items)} items ({dict(n)}) -> {path}")
        rep = balance_report(items)
        (out / f"{name}.balance.json").write_text(json.dumps(rep, indent=2))
        for k, v in rep.items():
            flag = "  <-- |SMD|>0.25, consider extra matching" if abs(v["smd"]) > 0.25 else ""
            print(f"    {k:<11} TA={v['TA']:.3f} TD={v['TD']:.3f} SMD={v['smd']:+.2f}{flag}")


if __name__ == "__main__":
    main()
