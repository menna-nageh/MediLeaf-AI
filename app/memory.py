"""
memory.py
---------
Short-term conversation memory for follow-up questions
("What are the side effects?" -> "Can children use it?").

Two distinct concerns are kept separate on purpose:

1. `ConversationMemory` - a small sliding window (default: last 3 turns)
   that is fed back into the LLM prompt so it can resolve follow-up
   questions about the same, single uploaded medicine. This is deliberately
   short: leaflet Q&A does not need long-range memory, and keeping it short
   reduces the chance of stale context leaking into a new question.

2. `SearchHistory` - an unbounded (per-session) log of every question asked,
   used only to populate the "previous questions" list in the sidebar so a
   user can click to reopen an earlier question. This is a UI convenience
   and is never sent to the LLM.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from app.config import settings


@dataclass
class Turn:
    """One question/answer exchange."""

    question: str
    answer: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class ConversationMemory:
    """Fixed-size sliding window of the most recent Q&A turns."""

    def __init__(self, window_size: int | None = None) -> None:
        self._window_size = window_size or settings.memory_window
        self._turns: deque[Turn] = deque(maxlen=self._window_size)

    def add_turn(self, question: str, answer: str) -> None:
        self._turns.append(Turn(question=question, answer=answer))

    def get_turns(self) -> list[Turn]:
        return list(self._turns)

    def as_prompt_context(self) -> str:
        """
        Render the memory window as plain text suitable for inclusion in a
        prompt, so the LLM can resolve pronouns/follow-ups
        ("it", "the medicine", "can children use it").
        """
        if not self._turns:
            return "No previous conversation in this session."
        lines = []
        for i, turn in enumerate(self._turns, start=1):
            lines.append(f"Q{i}: {turn.question}\nA{i}: {turn.answer}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()


class SearchHistory:
    """Unbounded, session-scoped log of every question asked (for the sidebar)."""

    def __init__(self) -> None:
        self._entries: list[str] = []

    def add(self, question: str) -> None:
        # Avoid piling up exact, back-to-back duplicates (e.g. double clicks).
        if not self._entries or self._entries[-1] != question:
            self._entries.append(question)

    def all(self) -> list[str]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()
