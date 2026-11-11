import threading

from .base import Method, render

MODEL_NAME = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
_compressor = None
_lock = threading.Lock()


def default_compressor():
    global _compressor
    with _lock:
        if _compressor is None:
            from llmlingua import PromptCompressor

            from ..core import config

            _compressor = PromptCompressor(model_name=MODEL_NAME, use_llmlingua2=True, device_map=config.MEMTRIM_ENCODER_DEVICE)
        return _compressor


class LLMLingua2(Method):
    name = "llmlingua2"

    def __init__(self, top_k=5, rate=0.5, compress=None, **params):
        super().__init__(top_k=top_k, **params)
        self.rate = rate
        self.compress = compress

    def _compress(self, texts):
        if self.compress is not None:
            return self.compress(texts, self.rate)
        result = default_compressor().compress_prompt(texts, rate=self.rate, force_tokens=["\n", "?"])
        return result["compressed_prompt"]

    def context(self, item, entries, llm):
        block = render(entries[: self.top_k])
        if block is None:
            return None
        return self._compress(block.split("\n")) or None
