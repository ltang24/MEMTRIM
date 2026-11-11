import importlib

BACKENDS = {
    "mem0": ("memtrim.memory.mem0", "Mem0Backend"),
    "amem": ("memtrim.memory.amem", "AMemBackend"),
    "memgpt": ("memtrim.memory.memgpt", "MemGPTBackend"),
    "graphiti": ("memtrim.memory.graphiti", "GraphitiBackend"),
}


def create(name, store_id, **kwargs):
    module_name, class_name = BACKENDS[name]
    return getattr(importlib.import_module(module_name), class_name)(store_id, **kwargs)
