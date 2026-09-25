"""Illustrate RQ1 metrics with synthetic predictions, not paper results."""

from __future__ import annotations

import json
from pathlib import Path


def load_examples(path: Path) -> list[dict[str, str]]:
    """Read the synthetic examples from a JSONL file."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compute_metrics(examples: list[dict[str, str]]) -> dict[str, int | float]:
    """Compute accuracies and transition rates using the total split size."""
    n = len(examples)
    if n == 0:
        raise ValueError("Cannot evaluate an empty split.")

    correctness = [
        (
            example["no_memory_prediction"] == example["gold_answer"],
            example["memory_prediction"] == example["gold_answer"],
        )
        for example in examples
    ]
    correct_to_wrong = sum(before and not after for before, after in correctness)
    wrong_to_correct = sum(not before and after for before, after in correctness)
    return {
        "n": n,
        "no_memory_accuracy": sum(before for before, _ in correctness) / n,
        "memory_accuracy": sum(after for _, after in correctness) / n,
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong_rate": correct_to_wrong / n,
        "wrong_to_correct_rate": wrong_to_correct / n,
    }


def main() -> None:
    data_path = Path(__file__).resolve().parents[1] / "data" / "rq1_toy.jsonl"
    examples = load_examples(data_path)
    split_metrics = {
        split: compute_metrics([example for example in examples if example["split"] == split])
        for split in ("mismatch", "match")
    }

    print("RQ1 minimal evaluation")
    print("Synthetic toy data; not paper results.")
    print()
    print(
        f"{'Split':<10} {'N':>3} {'No-memory Acc.':>16} {'Memory Acc.':>13} "
        f"{'✓->✗':>8} {'✗->✓':>8}"
    )
    for split, metrics in split_metrics.items():
        print(
            f"{split.capitalize():<10} {metrics['n']:>3} "
            f"{metrics['no_memory_accuracy']:>16.1%} "
            f"{metrics['memory_accuracy']:>13.1%} "
            f"{metrics['correct_to_wrong_rate']:>8.1%} "
            f"{metrics['wrong_to_correct_rate']:>8.1%}"
        )

    for split, metrics in split_metrics.items():
        print(f"\n{split.capitalize()}:")
        print(f"correct -> wrong: {metrics['correct_to_wrong']} / {metrics['n']}")
        print(f"wrong -> correct: {metrics['wrong_to_correct']} / {metrics['n']}")


if __name__ == "__main__":
    main()
