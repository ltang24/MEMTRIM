
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DS_ORDER = ["rippleedits", "counterlogic", "cladder"]
DS_NAME = {"rippleedits": "RippleEdits", "counterlogic": "CounterLogic", "cladder": "CLADDER-Derived"}
METHOD_NAME = {
    "no_memory": "No Mem.", "memory": "<backbone>", "facts_only": "Facts-only", "top1": "Top-1", "mmr": "MMR",
    "sim_filter": "Sim. filter", "recomp": "RECOMP", "llmlingua2": "LLMLingua-2",
    "context_faithful": "Context-faithful", "answer_first": "Answer-first", "micro_act": "Micro-Act",
    "memtrim": "MEMTRIM",
}
BACKEND_NAME = {"mem0": "Mem0", "amem": "A-Mem", "memgpt": "MemGPT", "graphiti": "Graphiti", "dense": "Dense"}


def load_preds(run_dir: str | Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(run_dir, "preds").glob("*.jsonl")):
        rows += [json.loads(l) for l in open(p)]
    if not rows:
        raise SystemExit(f"No predictions under {run_dir}/preds")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.method == "no_memory"][["id", "seed", "backend", "correct"]].rename(columns={"correct": "c0"})
    d = df.merge(base, on=["id", "seed", "backend"], how="left")
    d["c0"] = d["c0"].fillna(False).astype(bool)
    d["correct"] = d["correct"].astype(bool)
    per_seed = []
    for (ds, be, m, s), g in d.groupby(["dataset", "backend", "method", "seed"]):
        ta, td = g[g.condition == "TA"], g[g.condition == "TD"]
        per_seed.append(dict(
            dataset=ds, backend=be, method=m, seed=s,
            TA=100 * ta.correct.mean() if len(ta) else float("nan"),
            TD=100 * td.correct.mean() if len(td) else float("nan"),
            c2w=100 * (td.c0 & ~td.correct).mean() if len(td) else float("nan"),
            w2c=100 * (~td.c0 & td.correct).mean() if len(td) else float("nan"),
            n_TA=len(ta), n_TD=len(td),
        ))
    ps = pd.DataFrame(per_seed)
    agg = ps.groupby(["dataset", "backend", "method"]).agg(
        TA=("TA", "mean"), TA_sd=("TA", "std"), TD=("TD", "mean"), TD_sd=("TD", "std"),
        c2w=("c2w", "mean"), w2c=("w2c", "mean"), n_TA=("n_TA", "first"), n_TD=("n_TD", "first"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return agg.fillna({"TA_sd": 0.0, "TD_sd": 0.0})


def _fmt(m, sd):
    return "–" if pd.isna(m) else f"{m:.1f}±{sd:.1f}"


def to_markdown(agg: pd.DataFrame, methods: list[str] | None = None) -> str:
    out = []
    for be in sorted(agg.backend.unique()):
        sub = agg[agg.backend == be]
        ms = methods or [m for m in METHOD_NAME if m in set(sub.method)] + sorted(set(sub.method) - set(METHOD_NAME))
        dss = [d for d in DS_ORDER if d in set(sub.dataset)]
        head = "| Method | " + " | ".join(f"{DS_NAME[d]} TA | TD-Dup | ✓→✗ | ✗→✓" for d in dss) + " |"
        out += [f"\n### Backbone: {BACKEND_NAME.get(be, be)}\n", head, "|" + "---|" * (1 + 4 * len(dss))]
        for m in ms:
            cells = []
            for d in dss:
                r = sub[(sub.dataset == d) & (sub.method == m)]
                if r.empty:
                    cells += ["", "", "", ""]
                    continue
                r = r.iloc[0]
                nm = m == "no_memory"
                cells += [_fmt(r.TA, r.TA_sd), _fmt(r.TD, r.TD_sd),
                          "–" if nm else f"{r.c2w:.1f}", "–" if nm else f"{r.w2c:.1f}"]
            name = METHOD_NAME.get(m, m).replace("<backbone>", BACKEND_NAME.get(be, be))
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def to_latex(agg: pd.DataFrame) -> str:
    lines = []
    for be in sorted(agg.backend.unique()):
        sub = agg[agg.backend == be]
        lines.append(f"% Backbone: {BACKEND_NAME.get(be, be)}")
        for m in [m for m in METHOD_NAME if m in set(sub.method)]:
            cells = []
            for d in [d for d in DS_ORDER if d in set(sub.dataset)]:
                r = sub[(sub.dataset == d) & (sub.method == m)]
                if r.empty:
                    cells += ["", "", "", ""]
                    continue
                r = r.iloc[0]
                nm = m == "no_memory"
                cells += [f"{r.TA:.1f}$_{{\\pm{r.TA_sd:.1f}}}$", f"{r.TD:.1f}$_{{\\pm{r.TD_sd:.1f}}}$",
                          "–" if nm else f"{r.c2w:.1f}", "–" if nm else f"{r.w2c:.1f}"]
            name = METHOD_NAME[m].replace("<backbone>", BACKEND_NAME.get(be, be))
            lines.append(name + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def ablation_figure(agg: pd.DataFrame, out: Path) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    abl = ["memtrim-keep_shared", "memtrim-keep_conflicts", "memtrim-keep_old_answer",
           "memtrim-ignore_task_change", "memtrim-embedding_match"]
    rows = []
    for (ds, be), sub in agg.groupby(["dataset", "backend"]):
        full = sub[sub.method == "memtrim"]
        if full.empty:
            continue
        for a in abl:
            r = sub[sub.method == a]
            if not r.empty:
                rows.append(dict(dataset=ds, backend=be, ablation=a.split("-", 1)[1],
                                 delta=float(r.TD.iloc[0] - full.TD.iloc[0]), sd=float(r.TD_sd.iloc[0])))
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    combos = list(d.groupby(["backend", "dataset"]).groups)
    fig, ax = plt.subplots(figsize=(1.6 * len(combos) + 1, 3))
    w = 0.8 / len(abl)
    for j, a in enumerate([x.split("-", 1)[1] for x in abl]):
        vals = [d[(d.backend == b) & (d.dataset == s) & (d.ablation == a)].delta.sum() for b, s in combos]
        ax.bar([i + j * w for i in range(len(combos))], vals, w, label=a.replace("_", " "))
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(combos))])
    ax.set_xticklabels([f"{BACKEND_NAME.get(b, b)}\n{DS_NAME.get(s, s)}" for b, s in combos], fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Δ TD-Dup accuracy vs full (%)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig5_ablation.pdf")
    fig.savefig(out / "fig5_ablation.png", dpi=200)
    return d


def runtime_figure(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Figure 7: % change in total reasoning time, MEMTRIM vs backbone."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = df[df.method.isin(["memory", "memtrim"])].groupby(["dataset", "backend", "seed", "method"]).seconds.sum().unstack()
    t["pct"] = 100 * (t["memtrim"] - t["memory"]) / t["memory"]
    s = t.groupby(["dataset", "backend"]).pct.agg(["mean", "std"]).reset_index()
    dss = [d for d in DS_ORDER if d in set(s.dataset)]
    fig, axes = plt.subplots(1, len(dss), figsize=(3.2 * len(dss), 2.4), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        sub = s[s.dataset == ds]
        ax.barh([BACKEND_NAME.get(b, b) for b in sub.backend], sub["mean"], xerr=sub["std"].fillna(0))
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(DS_NAME[ds], fontsize=9)
        ax.set_xlabel("Δ reasoning time (%)")
    fig.tight_layout()
    fig.savefig(out / "fig7_runtime.pdf")
    fig.savefig(out / "fig7_runtime.png", dpi=200)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ablation", action="store_true", help="also draw Figure 5")
    ap.add_argument("--runtime", action="store_true", help="also draw Figure 7")
    a = ap.parse_args()
    run = Path(a.run_dir)
    df = load_preds(run)
    agg = summarize(df)
    agg.to_csv(run / "summary.csv", index=False)
    md = to_markdown(agg)
    (run / "tables.md").write_text(md)
    (run / "tables.tex").write_text(to_latex(agg))
    print(md)
    if a.ablation:
        print(ablation_figure(agg, run).to_string())
    if a.runtime:
        print(runtime_figure(df, run).to_string())


if __name__ == "__main__":
    main()
