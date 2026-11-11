from collections import defaultdict


def _accuracy(preds):
    if not preds:
        return None
    return 100.0 * sum(1 for p in preds if p["correct"]) / len(preds)


def _flip_rates(td_preds, no_memory_correct):
    scored = [p for p in td_preds if (p.get("dataset"), p["item_id"]) in no_memory_correct]
    if not scored:
        return None, None
    n = len(scored)
    base = [no_memory_correct[(p.get("dataset"), p["item_id"])] for p in scored]
    up = sum(1 for p, b in zip(scored, base) if not b and p["correct"])
    down = sum(1 for p, b in zip(scored, base) if b and not p["correct"])
    return 100.0 * up / n, 100.0 * down / n


def _ordered_unique(values):
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _row(dataset, memory_system, method, ta_preds, td_preds, flips):
    return {
        "dataset": dataset,
        "memory_system": memory_system,
        "method": method,
        "ta_acc": _accuracy(ta_preds),
        "td_acc": _accuracy(td_preds),
        "n_ta": len(ta_preds),
        "n_td": len(td_preds),
        "flip_up_pct": flips[0],
        "flip_down_pct": flips[1],
    }


def summarize(predictions):
    no_memory_correct = {(p.get("dataset"), p["item_id"]): p["correct"] for p in predictions if p["method"] == "no_memory"}
    groups = defaultdict(list)
    for p in predictions:
        groups[(p.get("dataset"), p["memory_system"], p["method"], p["condition"])].append(p)

    rows = []
    for dataset in _ordered_unique(p.get("dataset") for p in predictions):
        subset = [p for p in predictions if p.get("dataset") == dataset]
        no_mem_ta = [p for p in subset if p["method"] == "no_memory" and p["condition"] == "TA"]
        no_mem_td = [p for p in subset if p["method"] == "no_memory" and p["condition"] == "TD_DUP"]
        rows.append(_row(dataset, "none", "no_memory", no_mem_ta, no_mem_td, (None, None)))
        systems = sorted({p["memory_system"] for p in subset if p["memory_system"] and p["memory_system"] != "none"})
        methods = _ordered_unique(p["method"] for p in subset if p["method"] != "no_memory")
        for ms in systems:
            for method in methods:
                ta_preds = groups.get((dataset, ms, method, "TA"), [])
                td_preds = groups.get((dataset, ms, method, "TD_DUP"), [])
                if not ta_preds and not td_preds:
                    continue
                rows.append(_row(dataset, ms, method, ta_preds, td_preds, _flip_rates(td_preds, no_memory_correct)))
    return rows


def _fmt(x):
    return "-" if x is None else f"{x:.1f}"


def format_summary(rows):
    w = max([10] + [len(r["method"]) + 1 for r in rows])
    header = f"{'memory_system':<14}{'method':<{w}}{'TA_acc':>8}{'n_ta':>6}{'TD-Dup_acc':>12}{'n_td':>6}{'flip_wrong->correct':>20}{'flip_correct->wrong':>20}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['memory_system']:<14}{r['method']:<{w}}{_fmt(r['ta_acc']):>8}{r['n_ta']:>6}"
            f"{_fmt(r['td_acc']):>12}{r['n_td']:>6}{_fmt(r['flip_up_pct']):>20}{_fmt(r['flip_down_pct']):>20}"
        )
    return "\n".join(lines)


def print_summary(rows):
    print(format_summary(rows))
