import argparse
import json

from . import metrics


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def render(predictions):
    rows = metrics.summarize(predictions)
    sections = []
    for dataset in metrics._ordered_unique(r["dataset"] for r in rows):
        title = f"=== {dataset} ===" if dataset else "=== summary ==="
        sections.append(title + "\n" + metrics.format_summary([r for r in rows if r["dataset"] == dataset]))
    return "\n\n".join(sections)


def main():
    ap = argparse.ArgumentParser(prog="memtrim-report")
    ap.add_argument("predictions", nargs="+")
    args = ap.parse_args()
    predictions = [p for path in args.predictions for p in load_predictions(path)]
    print(render(predictions))


if __name__ == "__main__":
    main()
