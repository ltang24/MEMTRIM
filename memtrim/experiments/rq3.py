

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from ..report import BACKEND_NAME, DS_NAME
from ..runner import build_store, load_items, make_env, make_stores, run_query
from ..stats import mean_ci, quadratic, two_lines


def controlled_query(item: dict, rho: float, seed: int) -> tuple[str, str, list[str], int, int]:
    c = item["controlled"]
    units = c["units"]
    always = [i for i, u in enumerate(units) if u.get("always")]
    nonsup = [i for i, u in enumerate(units) if not u["support"] and not u.get("always")]
    sup = [i for i, u in enumerate(units) if u["support"]]
    random.Random(f"{item['id']}-{seed}").shuffle(nonsup)
    order = always + nonsup + sup
    n = len(units)
    k = max(int(round(rho * n)), len(always))
    included = set(order[:k])
    lines = []
    for i, u in enumerate(units):
        if i in included:
            t = u["mem_text"]
        elif u["support"]:
            t = u.get("td_text")
        else:
            t = None
        if t:
            lines.append(t)
    restored = all(i in included for i in sup)
    question = c["question_ta"] if restored else c["question_td"]
    gold, aliases = (c["ta_gold"], c["ta_aliases"]) if restored else (c["td_gold"], c["td_aliases"])
    body = " ".join(lines)
    text = (f"Context: {body}\n" if body else "") + f"Question: {question}"
    return text, gold, aliases, k, n


def run(cfg: dict) -> Path:
    env = make_env(cfg)
    levels = cfg.get("levels", [round(x, 1) for x in np.linspace(0, 1, 11)])
    methods = ["no_memory", "memory"]
    for dataset in cfg["datasets"]:
        items = [it for it in load_items(cfg["data_dir"], dataset, cfg.get("limit"), conditions=("TD",))
                 if it.get("controlled")]
        for be in cfg["memory"]["backends"]:
            for seed in cfg["seeds"]:
                env.llm.config.seed = seed
                path = env.out / "rq3" / f"{dataset}__{be}__seed{seed}.jsonl"
                if path.with_suffix(".done").exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w") as f:
                    for store_id, members in tqdm(make_stores(items, seed, int(cfg["memory"].get("store_size", 20))),
                                                  desc=f"rq3 {dataset}/{be}/s{seed}"):
                        backend, _, lookup = build_store(env, be, store_id, members, seed, None, None)
                        try:
                            for it in members:
                                for rho in levels:
                                    q, gold, aliases, k, n = controlled_query(it, rho, seed)
                                    item = dict(it, answer_type=it["controlled"]["answer_type"])
                                    for row in run_query(env, item, backend, None, lookup, None, methods,
                                                         query_text=q, gold=(gold, aliases)):
                                        row.update(backend=be, seed=seed, rho=rho, k_units=k, n_units=n)
                                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        finally:
                            backend.close()
                path.with_suffix(".done").write_text("ok")
    analyze(env.out)
    return env.out


def analyze(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(l) for p in sorted((out / "rq3").glob("*.jsonl")) for l in open(p)]
    if not rows:
        return
    by = defaultdict(dict)
    for r in rows:
        by[(r["dataset"], r["backend"], r["seed"], r["id"], r["rho"])][r["method"]] = r["correct"]
    delta = defaultdict(list)  # (ds, be) -> list of (rho, seed, d)
    for (ds, be, seed, _, rho), m in by.items():
        if "memory" in m and "no_memory" in m:
            delta[(ds, be)].append((rho, seed, float(m["memory"]) - float(m["no_memory"])))
    report = {}
    combos = sorted(delta)
    datasets = sorted({d for d, _ in combos})
    for ds in datasets:
        bes = [b for b in ("mem0", "amem", "memgpt", "graphiti", "dense") if (ds, b) in delta]
        fig, axes = plt.subplots(1, len(bes), figsize=(2.8 * len(bes), 2.5), sharey=True, squeeze=False)
        for ax, be in zip(axes[0], bes):
            arr = np.array(delta[(ds, be)])
            rhos = sorted(set(arr[:, 0]))
            means, cis = [], []
            for r in rhos:
                per_seed = [100 * arr[(arr[:, 0] == r) & (arr[:, 1] == s), 2].mean() for s in sorted(set(arr[:, 1]))]
                m, ci = mean_ci(per_seed)
                means.append(m)
                cis.append(ci)
            means, cis = np.array(means), np.array(cis)
            ax.plot(rhos, means, marker="o", ms=3)
            ax.fill_between(rhos, means - cis, means + cis, alpha=0.25)
            ax.axhline(0, ls="--", color="gray", lw=0.6)
            ax.set_title(BACKEND_NAME.get(be, be), fontsize=9)
            ax.set_xlabel("overlapping evidence")
            tl = two_lines(arr[:, 0], arr[:, 2], shape="u")
            qd = quadratic(arr[:, 0], arr[:, 2])
            report[f"{ds}/{be}"] = dict(levels=list(map(float, rhos)), delta_acc=means.tolist(), ci=cis.tolist(),
                                        two_lines=tl, quadratic=qd)
        axes[0][0].set_ylabel("Δ Accuracy (%)")
        fig.suptitle(DS_NAME.get(ds, ds), fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"fig_rq3_{ds}.pdf")
        fig.savefig(out / f"fig_rq3_{ds}.png", dpi=200)
    (out / "rq3_report.json").write_text(json.dumps(report, indent=2, default=float))
    for k, v in report.items():
        print(f"[{k}] two-lines U={v['two_lines'].get('ok')} quad b2={v['quadratic']['b2']:+.3f} (p={v['quadratic']['p']:.3g})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--analyze-only", default=None, help="results dir to re-analyze")
    a = ap.parse_args()
    if a.analyze_only:
        analyze(Path(a.analyze_only))
        return
    run(yaml.safe_load(open(a.config)))


if __name__ == "__main__":
    main()
