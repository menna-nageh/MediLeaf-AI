"""
memory.py
---------
Short-term conversation memory for MediLeaf AI.

The memory layer has two strictly separated responsibilities:

1. ConversationMemory
   - Stores a small sliding window of recent Q&A turns.
   - Helps resolve follow-up questions such as:
       "What are the side effects?"
       "Can children use it?"
       "What about pregnancy?"
   - NEVER acts as a source of medical evidence.
   - Retrieved leaflet chunks remain the only factual source.

2. SearchHistory
   - Stores questions asked during the current session.
   - Used only by the Streamlit UI.
   - NEVER sent to the LLM as factual context.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


from app.config import settings


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Turn:
    """
    Represents one question/answer exchange.

    The answer is stored for conversational continuity, but it must
    never be treated as retrieved evidence by the RAG pipeline.
    """

    question: str
    answer: str
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(
            timespec="seconds"
        )
    )


# ============================================================
# CONVERSATION MEMORY
# ============================================================

class ConversationMemory:
    """
    Fixed-size sliding window of recent conversation turns.

    Memory is intentionally short to reduce:
    - stale context
    - irrelevant information
    - prompt growth
    - cross-question contamination
    """

    def __init__(self, window_size: int | None = None) -> None:
        configured_size = (
            window_size
            if window_size is not None
            else settings.memory_window
        )

        if configured_size < 0:
            raise ValueError("Memory window size cannot be negative.")

        self._window_size = configured_size
        self._turns: deque[Turn] = deque(
            maxlen=self._window_size
        )

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    def add_turn(
        self,
        question: str,
        answer: str,
    ) -> None:
        """
        Add a completed Q&A turn to memory.
        """

        question = question.strip()
        answer = answer.strip()

        if not question:
            return

        self._turns.append(
            Turn(
                question=question,
                answer=answer,
            )
        )

    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------

    def get_turns(self) -> list[Turn]:
        """
        Return the current conversation turns.
        """

        return list(self._turns)

    def as_prompt_context(self) -> str:
        """
        Format recent conversation for the LLM.

        IMPORTANT:
        This context is only for resolving references and follow-up
        questions. It must never be considered medical evidence.
        """

        if not self._turns:
            return "No previous conversation in this session."

        lines = [
            "IMPORTANT: The following is conversation history only.",
            "It is NOT a source of medical evidence.",
            "Use it only to understand references such as 'it', "
            "'this medicine', or 'what about children'.",
            "",
        ]

        for index, turn in enumerate(self._turns, start=1):
            lines.append(
                f"Q{index}: {turn.question}\n"
                f"A{index}: {turn.answer}"
            )

        return "\n".join(lines)

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    def clear(self) -> None:
        """
        Remove all conversation turns.
        """

        self._turns.clear()

    def __len__(self) -> int:
        """
        Return the number of stored turns.
        """

        return len(self._turns)

    @property
    def window_size(self) -> int:
        """
        Return the configured maximum number of turns.
        """

        return self._window_size


# ============================================================
# SEARCH HISTORY
# ============================================================

class SearchHistory:
    """
    Session-scoped list of questions asked by the user.

    This exists exclusively for UI functionality and is never
    sent to the LLM as factual context.
    """

    def __init__(self) -> None:
        self._entries: list[str] = []

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    def add(self, question: str) -> None:
        """
        Add a question to the history.

        Consecutive duplicate questions are ignored to prevent
        accidental duplicate entries caused by double clicks.
        """

        question = question.strip()

        if not question:
            return

        if not self._entries or self._entries[-1] != question:
            self._entries.append(question)

    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------

    def all(self) -> list[str]:
        """
        Return all stored questions.
        """

        return list(self._entries)

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    def clear(self) -> None:
        """
        Clear the search history.
        """

        self._entries.clear()

    def __len__(self) -> int:
        """
        Return the number of stored questions.
        """

        return len(self._entries)


# ============================================================
# MEMORY SAFETY HELPERS
# ============================================================

def build_memory_context(
    memory: ConversationMemory | None,
) -> str:
    """
    Safely obtain conversation context.

    This helper keeps memory handling explicit at the RAG boundary.
    """

    if memory is None:
        return "No previous conversation in this session."

    return memory.as_prompt_context()