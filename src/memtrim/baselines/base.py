from ..core import prompts


def render(entries):
    block = "\n".join(e["text"] for e in entries if e.get("text"))
    return block or None


def answer(llm, item, memory_block=None, instruction=None, extra_sections=None, max_tokens=64):
    messages = prompts.build_answer_messages(item.query, memory_block, item.answer_type, instruction, extra_sections)
    return llm.chat(messages, max_tokens=max_tokens, temperature=0.0, seed=0)["content"]


class Method:
    name = "method"
    uses_memory = True
    instruction = None

    def __init__(self, top_k=5, **params):
        self.top_k = top_k
        self.params = params

    @property
    def pool_size(self):
        return self.top_k

    def for_store(self, backend, llm):
        return self

    def select(self, item, entries):
        return entries[: self.top_k]

    def context(self, item, entries, llm):
        return render(self.select(item, entries))

    def run(self, item, entries, llm):
        block = self.context(item, entries, llm)
        return {"output": answer(llm, item, block, instruction=self.instruction), "context": block}
