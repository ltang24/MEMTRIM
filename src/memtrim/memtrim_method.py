from concurrent.futures import ThreadPoolExecutor

from .baselines.base import Method
from .core import equivalence, parser
from .core.readtime import AblationFlags, build_memtrim_block
from .core.trie import Trie

_DEFAULT = object()


def _default_embed_fn():
    from .core.encoder import encode

    return encode


class MemtrimIndex:
    def __init__(self, llm, entries, embed_fn=None, workers=8):
        self.trie = Trie(
            embed_fn=embed_fn,
            equiv_key_fn=equivalence.make_key_equiv_fn(llm),
            equiv_value_fn=equivalence.make_value_equiv_fn(llm),
            equiv_task_fn=equivalence.make_task_equiv_fn(llm),
        )
        self.records = {}
        self.by_text = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            parsed = list(ex.map(lambda e: parser.parse_text(llm, e["text"]), entries))
        for entry, record in zip(entries, parsed):
            for i, unit in enumerate(record.evidence):
                self.trie.insert(unit, entry["entry_id"], record.support_mask[i])
            if record.outcome_evidence is not None:
                self.trie.insert(record.outcome_evidence, entry["entry_id"], False)
            if record.task is not None:
                self.trie.insert_task(record.task)
            self.records[entry["entry_id"]] = record
            self.by_text.setdefault(entry["text"], record)

    def record_for(self, entry):
        record = self.records.get(entry["entry_id"])
        return record if record is not None else self.by_text.get(entry["text"])


class MemtrimMethod(Method):
    name = "memtrim"

    def __init__(self, top_k=5, workers=8, embed_fn=_DEFAULT, index=None, **flags):
        super().__init__(top_k=top_k)
        self.workers = workers
        self.flags = AblationFlags(**flags)
        self.embed_fn = _default_embed_fn() if embed_fn is _DEFAULT else embed_fn
        self.index = index

    def for_store(self, backend, llm):
        index = MemtrimIndex(llm, backend.entries(), embed_fn=self.embed_fn, workers=self.workers)
        bound = MemtrimMethod(top_k=self.top_k, workers=self.workers, embed_fn=self.embed_fn, index=index)
        bound.flags = self.flags
        return bound

    def context(self, item, entries, llm):
        query_record = parser.parse_text(llm, item.query)
        records = [self.index.record_for(e) for e in entries[: self.top_k]]
        records = [r for r in records if r is not None and r.parse_ok]
        block = build_memtrim_block(query_record, records, self.index.trie, self.flags, embed_fn=self.embed_fn)
        return block or None
