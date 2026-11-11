from ..core import prompts
from .base import Method, answer


class AnswerFirst(Method):
    name = "answer_first"

    def run(self, item, entries, llm):
        initial = answer(llm, item, None)
        block = self.context(item, entries, llm)
        if block is None:
            return {"output": initial, "context": None, "initial": initial}
        output = answer(
            llm, item, block, instruction=prompts.ANSWER_FIRST_INSTRUCTION, extra_sections=[("INITIAL ANSWER", initial)]
        )
        return {"output": output, "context": block, "initial": initial}
