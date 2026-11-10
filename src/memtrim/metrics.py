from collections import defaultdict


def _accuracy(preds):
    if not preds:
        return None
    return 100.0 * sum(1 for p in preds if p["correct"]) / len(preds)


def _flip_rates(td_preds, no_memory_correct):
    scored = [p for p in td_preds if p["item_id"] in no_memory_correct]
    if not scored:
        return None, None
    n = len(scored)
    up = sum(1 for p in scored if not no_memory_correct[p["item_id"]] and p["correct"])
    down = sum(1 for p in scored if no_memory_correct[p["item_id"]] and not p["correct"])
    return 100.0 * up / n, 100.0 * down / n


def summarize(predictions):
    no_memory_correct = {p["item_id"]: p["correct"] for p in predictions if p["method"] == "no_memory"}

    by_system_method_condition = defaultdict(list)
    for p in predictions:
        by_system_method_condition[(p["memory_system"], p["method"], p["condition"])].append(p)

    memory_systems = sorted({p["memory_system"] for p in predictions if p["memory_system"] and p["memory_system"] != "none"})

    rows = []
    no_mem_ta = [p for p in predictions if p["method"] == "no_memory" and p["condition"] == "TA"]
    no_mem_td = [p for p in predictions if p["method"] == "no_memory" and p["condition"] == "TD_DUP"]
    rows.append(
        {
            "memory_system": "none",
            "method": "no_memory",
            "ta_acc": _accuracy(no_mem_ta),
            "td_acc": _accuracy(no_mem_td),
            "n_ta": len(no_mem_ta),
            "n_td": len(no_mem_td),
            "flip_up_pct": None,
            "flip_down_pct": None,
        }
    )

    for ms in memory_systems:
        for method in ("base", "memtrim"):
            ta_preds = by_system_method_condition.get((ms, method, "TA"), [])
            td_preds = by_system_method_condition.get((ms, method, "TD_DUP"), [])
            flip_up, flip_down = _flip_rates(td_preds, no_memory_correct)
            rows.append(
                {
                    "memory_system": ms,
                    "method": method,
                    "ta_acc": _accuracy(ta_preds),
                    "td_acc": _accuracy(td_preds),
                    "n_ta": len(ta_preds),
                    "n_td": len(td_preds),
                    "flip_up_pct": flip_up,
                    "flip_down_pct": flip_down,
                }
            )
    return rows


def _fmt(x):
    return "-" if x is None else f"{x:.1f}"


def print_summary(rows):
    header = f"{'memory_system':<14}{'method':<10}{'TA_acc':>8}{'n_ta':>6}{'TD-Dup_acc':>12}{'n_td':>6}{'flip_wrong->correct':>20}{'flip_correct->wrong':>20}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['memory_system']:<14}{r['method']:<10}{_fmt(r['ta_acc']):>8}{r['n_ta']:>6}"
            f"{_fmt(r['td_acc']):>12}{r['n_td']:>6}{_fmt(r['flip_up_pct']):>20}{_fmt(r['flip_down_pct']):>20}"
        )
