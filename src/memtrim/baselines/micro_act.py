from ..core import prompts
from .base import Method, answer


class MicroAct(Method):
    name = "micro_act"

    def run(self, item, entries, llm):
        block = self.context(item, entries, llm)
        if block is None:
            return {"output": answer(llm, item, None), "context": None, "analysis": None}
        messages = prompts.build_micro_act_analysis_messages(item.query, block)
        analysis = llm.chat(messages, max_tokens=512, temperature=0.0, seed=0)["content"]
        output = answer(llm, item, block, instruction=prompts.MICRO_ACT_INSTRUCTION, extra_sections=[("COMPARISON", analysis)])
        return {"output": output, "context": block, "analysis": analysis}
