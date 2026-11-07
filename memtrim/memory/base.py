"""Common interface over the four memory systems in §2.

Every backend stores complete interactions through its own write path and
retrieves with its own mechanism. Each returned item is linked back to the
interaction it came from (``interaction_id``) so that MEMTRIM - which keeps a
separate, write-time evidence index keyed by interaction - can process exactly
the memories the base system retrieved (App. A.1: the trie is never used as an
additional retriever).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Retrieved:
    text: str  # what the base system returns (entry / note / passage / fact)
    interaction_id: str | None  # source interaction; None if unknown
    score: float = 0.0
    rank: int = 0


def interaction_text(text: str, answer: str) -> str:
    return f"{text}\nAnswer: {answer}"


class MemoryBackend(ABC):
    name = "base"
    # Whether the base system keeps trustworthy write timestamps (App. A.1).
    has_timestamps = False

    @abstractmethod
    def add(self, interaction_id: str, text: str, answer: str, timestamp: float | None = None) -> None:
        """Write one completed interaction (x_m, y_m)."""

    @abstractmethod
    def search(self, query: str, k: int) -> list[Retrieved]:
        """Return up to k items, best first."""

    def add_raw(self, interaction_id: str, content: str, timestamp: float | None = None) -> None:
        """Write arbitrary content (used by memory-injection attacks)."""
        self.add(interaction_id, content, "", timestamp)

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


def dedupe_by_interaction(items: list[Retrieved], k: int) -> list[Retrieved]:
    """Keep the best-ranked item per interaction, re-rank, cut to k."""
    seen: set[str] = set()
    out: list[Retrieved] = []
    for it in items:
        key = it.interaction_id or f"__anon{len(out)}"
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= k:
            break
    for i, it in enumerate(out):
        it.rank = i
    return out
