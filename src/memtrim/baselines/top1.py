from .base import Method


class Top1(Method):
    name = "top1"

    def select(self, item, entries):
        return entries[:1]
