import numpy as np

from .base import Method


class SimFilter(Method):
    name = "sim_filter"

    def __init__(self, top_k=5, threshold=0.5, encode=None, **params):
        super().__init__(top_k=top_k, **params)
        self.threshold = threshold
        self.encode = encode

    def select(self, item, entries):
        entries = entries[: self.top_k]
        if not entries:
            return []
        if self.encode is None:
            from ..core.encoder import encode

            self.encode = encode
        vectors = np.asarray(self.encode([e["text"] for e in entries]))
        sims = vectors @ np.asarray(self.encode(item.query))
        return [e for e, s in zip(entries, sims) if float(s) >= self.threshold]
