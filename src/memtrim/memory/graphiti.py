import asyncio
import os
import re
import shutil
from datetime import datetime, timedelta, timezone

from ..core import config
from .base import MemoryBackend, make_entry

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

_BASE_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _local_embedder():
    from graphiti_core.embedder.client import EmbedderClient

    from ..core.encoder import encode

    class LocalEmbedder(EmbedderClient):
        async def create(self, input_data):
            text = input_data if isinstance(input_data, str) else " ".join(str(x) for x in input_data)
            return encode(text).tolist()

        async def create_batch(self, input_data_list):
            return [v.tolist() for v in encode(list(input_data_list))]

    return LocalEmbedder()


def _pass_through_reranker():
    from graphiti_core.cross_encoder.client import CrossEncoderClient

    class PassThroughReranker(CrossEncoderClient):
        async def rank(self, query, passages):
            n = max(len(passages), 1)
            return [(p, 1.0 - i / n) for i, p in enumerate(passages)]

    return PassThroughReranker()


def _llm_client(max_tokens):
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    llm_config = LLMConfig(
        api_key=config.FAKE_API_KEY,
        model=config.MEMTRIM_MODEL,
        small_model=config.MEMTRIM_MODEL,
        base_url=config.MEMTRIM_BASE_URL,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return OpenAIGenericClient(config=llm_config, max_tokens=max_tokens)


class GraphitiBackend(MemoryBackend):
    name = "graphiti"

    def __init__(self, store_id, uri=None, user=None, password=None, max_tokens=4096, **kwargs):
        super().__init__(store_id)
        from graphiti_core import Graphiti

        self.group_id = re.sub(r"[^A-Za-z0-9_-]", "_", store_id)
        self.loop = asyncio.new_event_loop()
        uri = uri or config.MEMTRIM_GRAPHITI_URI
        self.driver = self._neo4j(uri, user, password) if uri else self._kuzu()
        self.graphiti = Graphiti(
            graph_driver=self.driver,
            llm_client=_llm_client(max_tokens),
            embedder=_local_embedder(),
            cross_encoder=_pass_through_reranker(),
        )
        self._run(self._build_indices(bool(uri)))
        self.episode_owner = {}
        self.n_added = 0

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def _kuzu(self):
        from graphiti_core.driver.kuzu_driver import KuzuDriver

        os.makedirs(config.GRAPHITI_STORE_DIR, exist_ok=True)
        path = os.path.join(config.GRAPHITI_STORE_DIR, f"{self.group_id}.kuzu")
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
        return KuzuDriver(db=path)

    def _neo4j(self, uri, user, password):
        from graphiti_core.driver.neo4j_driver import Neo4jDriver

        return Neo4jDriver(uri, user or config.MEMTRIM_GRAPHITI_USER, password or config.MEMTRIM_GRAPHITI_PASSWORD)

    async def _build_indices(self, is_neo4j):
        if not is_neo4j:
            from graphiti_core.driver.driver import GraphProvider
            from graphiti_core.graph_queries import get_fulltext_indices

            for query in get_fulltext_indices(GraphProvider.KUZU):
                await self.driver.execute_query(query)
        await self.graphiti.build_indices_and_constraints()

    def add(self, interaction_id, text, answer, item_id=None, group=None, is_primary=True):
        from graphiti_core.nodes import EpisodeType

        result = self._run(
            self.graphiti.add_episode(
                name=interaction_id,
                episode_body=self.content(text, answer),
                source=EpisodeType.text,
                source_description="interaction",
                reference_time=_BASE_TIME + timedelta(minutes=self.n_added),
                group_id=self.group_id,
            )
        )
        self.episode_owner[result.episode.uuid] = interaction_id
        self.n_added += 1

    def _owner(self, edge):
        for episode_uuid in edge.episodes or []:
            if episode_uuid in self.episode_owner:
                return self.episode_owner[episode_uuid]
        return None

    def search(self, query, k):
        edges = self._run(self.graphiti.search(query, group_ids=[self.group_id], num_results=k))
        return [make_entry(e.uuid, e.fact, self._owner(e)) for e in edges[:k]]

    def entries(self):
        from graphiti_core.edges import EntityEdge
        from graphiti_core.errors import GroupsEdgesNotFoundError

        try:
            edges = self._run(EntityEdge.get_by_group_ids(self.driver, [self.group_id]))
        except GroupsEdgesNotFoundError:
            return []
        edges = sorted(edges, key=lambda e: (e.created_at, e.uuid))
        return [make_entry(e.uuid, e.fact, self._owner(e)) for e in edges]

    def close(self):
        self._run(self.graphiti.close())
        self.loop.close()
