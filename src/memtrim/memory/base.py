from ..datasets.base import compose


def make_entry(entry_id, text, interaction_id, score=None):
    return {"entry_id": entry_id, "text": text, "interaction_id": interaction_id, "score": score}


class MemoryBackend:
    name = "base"

    def __init__(self, store_id, **kwargs):
        self.store_id = store_id

    def content(self, text, answer):
        return compose(text, answer)

    def add(self, interaction_id, text, answer, item_id=None, group=None, is_primary=True):
        raise NotImplementedError

    def search(self, query, k):
        raise NotImplementedError

    def entries(self):
        raise NotImplementedError

    def close(self):
        pass
