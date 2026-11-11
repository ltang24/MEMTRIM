from .base import Method


class FactsOnly(Method):
    name = "facts_only"

    def select(self, item, entries):
        seen, out = set(), []
        for e in entries[: self.top_k]:
            source = e.get("source_text")
            if not source or e.get("interaction_id") in seen:
                continue
            seen.add(e.get("interaction_id"))
            out.append({"text": source})
        return out
