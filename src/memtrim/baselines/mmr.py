import numpy as np

from .base import Method


class MMR(Method):
    name = "mmr"

    def __init__(self, top_k=5, pool=15, lambda_=0.5, encode=None, **params):
        super().__init__(top_k=top_k, **params)
        self.pool = pool
        self.lambda_ = lambda_
        self.encode = encode

    @property
    def pool_size(self):
        return max(self.pool, self.top_k)

    def _encode(self, texts):
        if self.encode is None:
            from ..core.encoder import encode

            self.encode = encode
        return self.encode(texts)

    def select(self, item, entries):
        pool = entries[: self.pool]
        if not pool:
            return []
        vectors = np.asarray(self._encode([e["text"] for e in pool]))
        relevance = vectors @ np.asarray(self._encode(item.query))
        selected, remaining = [], list(range(len(pool)))
        while remaining and len(selected) < self.top_k:
            def score(i):
                redundancy = max((float(vectors[i] @ vectors[j]) for j in selected), default=0.0)
                return self.lambda_ * float(relevance[i]) - (1 - self.lambda_) * redundancy

            best = max(remaining, key=score)
            selected.append(best)
            remaining.remove(best)
        return [pool[i] for i in selected]
