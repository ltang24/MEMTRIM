import uuid

from ..core import config
from .base import MemoryBackend, make_entry

CORE_LABEL = "recent_interaction"
TAG_PREFIX = "interaction:"


class MemGPTBackend(MemoryBackend):
    name = "memgpt"

    def __init__(self, store_id, base_url=None, model=None, embedding=None, **kwargs):
        super().__init__(store_id)
        from letta_client import Letta

        self.client = Letta(base_url=base_url or config.MEMTRIM_LETTA_BASE_URL)
        agent = self.client.agents.create(
            name=f"memtrim-{store_id}-{uuid.uuid4().hex[:8]}",
            model=model or config.MEMTRIM_LETTA_MODEL,
            embedding=embedding or config.MEMTRIM_LETTA_EMBEDDING,
            memory_blocks=[{"label": CORE_LABEL, "value": ""}],
        )
        self.agent_id = agent.id
        self.latest_interaction_id = None
        self.passage_owner = {}

    def add(self, interaction_id, text, answer, item_id=None, group=None, is_primary=True):
        content = self.content(text, answer)
        created = self.client.agents.passages.create(agent_id=self.agent_id, text=content, tags=[TAG_PREFIX + interaction_id])
        for passage in created:
            self.passage_owner[passage.id] = interaction_id
        self.client.agents.blocks.update(CORE_LABEL, agent_id=self.agent_id, value=content)
        self.latest_interaction_id = interaction_id

    def _owner(self, passage_id, tags):
        for tag in tags or []:
            if tag.startswith(TAG_PREFIX):
                return tag[len(TAG_PREFIX):]
        return self.passage_owner.get(passage_id)

    def search(self, query, k):
        results = []
        block = self.client.agents.blocks.retrieve(CORE_LABEL, agent_id=self.agent_id)
        if block.value:
            results.append(make_entry(f"core:{self.latest_interaction_id}", block.value, self.latest_interaction_id))
        if len(results) < k:
            response = self.client.agents.passages.search(agent_id=self.agent_id, query=query, top_k=k)
            for hit in response.results:
                owner = self._owner(hit.id, hit.tags)
                if owner == self.latest_interaction_id and results:
                    continue
                results.append(make_entry(hit.id, hit.content, owner))
                if len(results) >= k:
                    break
        return results

    def entries(self):
        passages = self.client.agents.passages.list(agent_id=self.agent_id, limit=1000, ascending=True)
        return [make_entry(p.id, p.text, self._owner(p.id, p.tags)) for p in passages]

    def close(self):
        self.client.agents.delete(self.agent_id)
