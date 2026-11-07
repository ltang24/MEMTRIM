
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memtrim.runner import run_main  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="items per condition (overrides config)")
    ap.add_argument("--run-name", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.limit is not None:
        cfg["limit"] = a.limit
    if a.run_name:
        cfg["run_name"] = a.run_name
    out = run_main(cfg)
    print(f"Predictions in {out}/preds. Make tables with:\n  python -m memtrim.report {out}")


if __name__ == "__main__":
    main()
