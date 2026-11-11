import json
import threading
from collections import defaultdict

from . import config

_lock = threading.Lock()
_counts = defaultdict(int)


def record_failure(system, stage, item_id, store_id, error):
    with _lock:
        _counts[(system, stage)] += 1
        with open(config.ERROR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "system": system,
                        "stage": stage,
                        "item_id": item_id,
                        "store_id": store_id,
                        "error": repr(error),
                    },
                    ensure_ascii=False,
                )
            )
            f.write("\n")


def counts():
    with _lock:
        return dict(_counts)


def reset():
    with _lock:
        _counts.clear()
