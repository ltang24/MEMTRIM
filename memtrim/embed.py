"""Sentence encoder wrapper.

``sentence-transformers/all-mpnet-base-v2`` is the encoder named in Appendix C.
The ``hash`` encoder is a dependency-free bag-of-words fallback used ONLY for
offline smoke tests; do not use it for reported numbers.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Encoder:
    def __init__(self, name: str = "sentence-transformers/all-mpnet-base-v2", device: str | None = None):
        self.name = name
        self._model = None
        if name != "hash":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(name, device=device)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Return L2-normalised embeddings, shape (n, d)."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        if self._model is None:
            return np.stack([_hash_vec(t) for t in texts])
        emb = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(emb, dtype=np.float32)

    def sim(self, a: list[str], b: list[str]) -> np.ndarray:
        return self.encode(a) @ self.encode(b).T


@lru_cache(maxsize=200_000)
def _hash_vec(text: str, dim: int = 384) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for tok in _TOKEN.findall(text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


_ENCODERS: dict[str, Encoder] = {}


def get_encoder(name: str) -> Encoder:
    if name not in _ENCODERS:
        _ENCODERS[name] = Encoder(name)
    return _ENCODERS[name]
