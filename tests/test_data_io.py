import os
import tempfile
import unittest

from memtrim.datasets import registry
from memtrim.datasets.base import TA, TD_DUP, Item, build_stores


def toy_items():
    return [
        Item(id=f"x{i}_{c}", pair_id=f"x{i}", dataset="toy", condition=c, group="g", query=f"q{i}", gold="yes",
             answer_type="yes_no", memory={"text": f"m{i}", "answer": "no"})
        for i in range(3) for c in (TA, TD_DUP)
    ]


class DataFolderTest(unittest.TestCase):
    def test_round_trip_and_settings_mismatch(self):
        items = toy_items()
        stores = build_stores(items, 2)
        meta = registry.settings("toy", 1, 3, 2, {"b": 1, "a": 2})
        with tempfile.TemporaryDirectory() as d:
            registry.save("toy", meta, items, stores, root=d)
            loaded_items, loaded_stores = registry.load("toy", meta, root=d)
            self.assertEqual([it.to_dict() for it in loaded_items], [it.to_dict() for it in items])
            self.assertEqual({k: [ia.to_dict() for ia in v] for k, v in loaded_stores.items()},
                             {k: [ia.to_dict() for ia in v] for k, v in stores.items()})
            with open(os.path.join(d, "toy", "items.json"), "rb") as f:
                first = f.read()
            registry.save("toy", meta, loaded_items, loaded_stores, root=d)
            with open(os.path.join(d, "toy", "items.json"), "rb") as f:
                self.assertEqual(f.read(), first)
            self.assertIsNone(registry.load("toy", registry.settings("toy", 2, 3, 2, {"a": 2, "b": 1}), root=d))
            self.assertIsNone(registry.load("other", meta, root=d))


if __name__ == "__main__":
    unittest.main()
