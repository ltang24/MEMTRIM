import os
import tempfile

import chromadb
from chromadb.config import Settings
from agentic_memory import retrievers
from agentic_memory.memory_system import AgenticMemorySystem

from . import config, errors

os.environ.setdefault("OPENAI_BASE_URL", config.MEMTRIM_BASE_URL)
os.environ.setdefault("OPENAI_API_KEY", config.FAKE_API_KEY)


def _isolated_chroma_retriever_init(self, collection_name="memories", model_name="all-MiniLM-L6-v2"):
    persist_dir = tempfile.mkdtemp(prefix="amem_chroma_", dir=config.MEM0_STORE_DIR)
    self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
    self.embedding_function = retrievers.SentenceTransformerEmbeddingFunction(model_name=model_name)
    self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.embedding_function)


retrievers.ChromaRetriever.__init__ = _isolated_chroma_retriever_init


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


def build_store(store_id, interactions):
    system = AgenticMemorySystem(
        model_name=config.ENCODER_MODEL_NAME,
        llm_backend="openai",
        llm_model=config.MEMTRIM_MODEL,
        api_key=config.FAKE_API_KEY,
    )
    id_to_interaction = {}
    fragments = []
    for interaction in interactions:
        note_id = _add_note_with_retry(system, store_id, interaction)
        if note_id is None:
            continue
        id_to_interaction[note_id] = interaction
        fragments.append(
            {
                "id": note_id,
                "text": interaction.text,
                "interaction_id": interaction.interaction_id,
                "item_id": interaction.item_id,
            }
        )
    return system, id_to_interaction, fragments


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
