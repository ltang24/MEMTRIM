from ..core import prompts
from .base import Method


class ContextFaithful(Method):
    name = "context_faithful"
    instruction = prompts.CONTEXT_FAITHFUL_INSTRUCTION
