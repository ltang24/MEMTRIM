

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..report import BACKEND_NAME, DS_NAME, load_preds
from ..stats import overlap_tests


def collect(df, dataset: str):
    d = df[(df.dataset == dataset) & (df.condition == "TD")]
    base = d[d.method == "no_memory"].set_index(["id", "seed", "backend"]).correct
    mem = d[d.method == "memory"].set_index(["id", "seed", "backend"])
    j = mem.join(base.rename("c0"), how="inner")
    j = j[j.c0.astype(bool) & j.overlap_top1.notna()]
    out = {}
    for be, g in j.groupby(level="backend"):
        out[be] = (g[g.correct.astype(bool)].overlap_top1.tolist(), g[~g.correct.astype(bool)].overlap_top1.tolist())
    return out


def plot(groups: dict, tests: dict, path: Path, dataset: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bes = [b for b in ("mem0", "amem", "memgpt", "graphiti", "dense") if b in groups]
    fig, axes = plt.subplots(1, len(bes), figsize=(2.6 * len(bes), 2.6), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, be in zip(axes[0], bes):
        s, f = groups[be]
        data = [s, f]
        parts = ax.violinplot([x if x else [np.nan] for x in data], positions=[0, 1], showextrema=False)
        for body, col in zip(parts["bodies"], ("#4C72B0", "#C44E52")):
            body.set_facecolor(col)
            body.set_alpha(0.35)
        ax.boxplot([x if x else [np.nan] for x in data], positions=[0, 1], widths=0.15, showfliers=False)
        for i, (x, col) in enumerate(zip(data, ("#4C72B0", "#C44E52"))):
            ax.scatter(i - 0.3 + rng.uniform(-0.05, 0.05, len(x)), x, s=2, alpha=0.4, color=col)
        t = tests[be]
        p = t.get("centrality_p")
        ax.set_title(f"{BACKEND_NAME.get(be, be)}\n" + (f"centrality p={p:.3g}" if p is not None else "n/a"), fontsize=8)
        ax.set_xticks([0, 1], ["Success", "Failure"])
    axes[0][0].set_ylabel("Query–memory overlap")
    fig.suptitle(DS_NAME.get(dataset, dataset), fontsize=9)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--dataset", default=None)
    a = ap.parse_args()
    run = Path(a.run_dir)
    df = load_preds(run)
    for ds in [a.dataset] if a.dataset else sorted(df.dataset.unique()):
        groups = collect(df, ds)
        if not groups:
            print(f"[{ds}] no TD-Dup cases that are correct without memory; skipping")
            continue
        tests = {be: overlap_tests(*v) for be, v in groups.items()}
        (run / f"rq2_{ds}.json").write_text(json.dumps(tests, indent=2, default=float))
        plot(groups, tests, run / f"fig_rq2_{ds}", ds)
        for be, t in tests.items():
            print(f"[{ds}/{be}] success={t['n_success']} failure={t['n_failure']} "
                  f"MWU p={t.get('mwu_p', float('nan')):.3g} centrality p={t.get('centrality_p', float('nan')):.3g} "
                  f"two-lines(inverted U)={t.get('two_lines', {}).get('ok')}")


if __name__ == "__main__":
    main()
