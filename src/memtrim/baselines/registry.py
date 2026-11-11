import importlib

METHODS = {
    "no_memory": ("memtrim.baselines.no_memory", "NoMemory"),
    "base_memory": ("memtrim.baselines.base_memory", "BaseMemory"),
    "facts_only": ("memtrim.baselines.facts_only", "FactsOnly"),
    "top1": ("memtrim.baselines.top1", "Top1"),
    "mmr": ("memtrim.baselines.mmr", "MMR"),
    "sim_filter": ("memtrim.baselines.sim_filter", "SimFilter"),
    "recomp": ("memtrim.baselines.recomp", "Recomp"),
    "llmlingua2": ("memtrim.baselines.llmlingua2", "LLMLingua2"),
    "context_faithful": ("memtrim.baselines.context_faithful", "ContextFaithful"),
    "answer_first": ("memtrim.baselines.answer_first", "AnswerFirst"),
    "micro_act": ("memtrim.baselines.micro_act", "MicroAct"),
    "memtrim": ("memtrim.memtrim_method", "MemtrimMethod"),
}


def create(name, **params):
    module_name, class_name = METHODS[name]
    return getattr(importlib.import_module(module_name), class_name)(**params)
