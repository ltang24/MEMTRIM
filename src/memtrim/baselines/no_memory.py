from .base import Method


class NoMemory(Method):
    name = "no_memory"
    uses_memory = False

    def context(self, item, entries, llm):
        return None
