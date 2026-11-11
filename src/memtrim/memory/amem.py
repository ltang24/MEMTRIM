import os
import tempfile
from types import SimpleNamespace

from ..core import config, errors
from .base import MemoryBackend, make_entry

os.environ.setdefault("OPENAI_BASE_URL", config.MEMTRIM_BASE_URL)
os.environ.setdefault("OPENAI_API_KEY", config.FAKE_API_KEY)

_patched = False


def _isolated_chroma_retriever_init(self, collection_name="memories", model_name="all-MiniLM-L6-v2"):
    import chromadb
    from agentic_memory import retrievers
    from chromadb.config import Settings

    persist_dir = tempfile.mkdtemp(prefix="amem_chroma_", dir=config.MEM0_STORE_DIR)
    self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
    self.embedding_function = retrievers.SentenceTransformerEmbeddingFunction(model_name=model_name)
    self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.embedding_function)


def _patch_amem():
    global _patched
    if _patched:
        return
    from agentic_memory import retrievers

    retrievers.ChromaRetriever.__init__ = _isolated_chroma_retriever_init
    _patched = True


def _add_note_with_retry(system, store_id, interaction):
    last_error = None
    for _ in range(2):
        try:
            analysis = system.analyze_content(interaction.text)
            return system.add_note(
                interaction.text,
                keywords=analysis.get("keywords", []),
                context=analysis.get("context", "General"),
                tags=analysis.get("tags", []),
                category=interaction.relation,
            )
        except Exception as e:
            last_error = e
    errors.record_failure("amem", "write", interaction.item_id, store_id, last_error)
    return None


def open_store():
    _patch_amem()
    from agentic_memory.memory_system import AgenticMemorySystem

    return AgenticMemorySystem(
        model_name=config.ENCODER_MODEL_NAME,
        llm_backend="openai",
        llm_model=config.MEMTRIM_MODEL,
        api_key=config.FAKE_API_KEY,
    )


def add_interaction(system, store_id, interaction, id_to_interaction):
    note_id = _add_note_with_retry(system, store_id, interaction)
    if note_id is None:
        return None
    id_to_interaction[note_id] = interaction
    return {
        "id": note_id,
        "text": interaction.text,
        "interaction_id": interaction.interaction_id,
        "item_id": interaction.item_id,
    }


def retrieve(system, id_to_interaction, query_text, top_k, item_id=None, store_id=None):
    last_error = None
    for _ in range(2):
        try:
            results = system.search_agentic(query_text, k=top_k)
            entries = []
            for r in results:
                interaction = id_to_interaction.get(r["id"])
                entries.append(
                    {
                        "id": r["id"],
                        "text": r["content"],
                        "interaction_id": interaction.interaction_id if interaction else None,
                        "item_id": interaction.item_id if interaction else None,
                        "score": r.get("score"),
                    }
                )
            return entries
        except Exception as e:
            last_error = e
    errors.record_failure("amem", "retrieve", item_id, store_id, last_error)
    return []


class AMemBackend(MemoryBackend):
    name = "amem"

    def __init__(self, store_id, **kwargs):
        super().__init__(store_id)
        self.system = open_store()
        self.id_map = {}
        self.fragments = []

    def add(self, interaction_id, text, answer, item_id=None, group=None, is_primary=True):
        interaction = SimpleNamespace(
            interaction_id=interaction_id, text=self.content(text, answer), item_id=item_id, is_primary=is_primary, relation=group
        )
        fragment = add_interaction(self.system, self.store_id, interaction, self.id_map)
        if fragment is not None:
            self.fragments.append(fragment)

    def search(self, query, k):
        hits = retrieve(self.system, self.id_map, query, k, store_id=self.store_id)
        return [make_entry(h["id"], h["text"], h["interaction_id"], h["score"]) for h in hits]

    def entries(self):
        return [make_entry(f["id"], f["text"], f["interaction_id"]) for f in self.fragments]

    def close(self):
        self.system = None
