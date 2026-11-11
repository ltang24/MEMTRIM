import re
import threading

from .base import Method

MODEL_NAME = "fangyuan/nq_extractive_compressor"
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_scorer = None
_lock = threading.Lock()


def split_sentences(text):
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


class ExtractiveScorer:
    def __init__(self, model_name=MODEL_NAME):
        import torch
        from transformers import AutoModel, AutoTokenizer

        from ..core import config

        self.torch = torch
        self.device = config.MEMTRIM_ENCODER_DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def _embed(self, texts):
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1)

    def __call__(self, query, sentences):
        q = self._embed([query])
        s = self._embed(sentences)
        return (s @ q.T).squeeze(-1).tolist()


def default_scorer():
    global _scorer
    with _lock:
        if _scorer is None:
            _scorer = ExtractiveScorer()
        return _scorer


class Recomp(Method):
    name = "recomp"

    def __init__(self, top_k=5, n_sentences=3, scorer=None, **params):
        super().__init__(top_k=top_k, **params)
        self.n_sentences = n_sentences
        self.scorer = scorer

    def context(self, item, entries, llm):
        sentences = [s for e in entries[: self.top_k] for s in split_sentences(e["text"])]
        if not sentences:
            return None
        scorer = self.scorer or default_scorer()
        scores = scorer(item.query, sentences)
        ranked = sorted(range(len(sentences)), key=lambda i: -scores[i])[: self.n_sentences]
        return "\n".join(sentences[i] for i in ranked)
