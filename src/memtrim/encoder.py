import threading

from sentence_transformers import SentenceTransformer

from . import config

_model = None
_lock = threading.Lock()


def get_encoder():
    global _model
    with _lock:
        if _model is None:
            _model = SentenceTransformer(config.ENCODER_MODEL_NAME, device=config.MEMTRIM_ENCODER_DEVICE)
        return _model


def encode(texts):
    model = get_encoder()
    single = isinstance(texts, str)
    inputs = [texts] if single else list(texts)
    vectors = model.encode(inputs, convert_to_numpy=True, normalize_embeddings=True)
    return vectors[0] if single else vectors
