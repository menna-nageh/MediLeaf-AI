"""
emergency.py
------------
Lightweight deterministic emergency keyword detection.

This module:
- Scans only the user's question.
- Uses deterministic keyword/phrase matching.
- Does NOT call an LLM.
- Does NOT diagnose the user.
- Does NOT generate medical advice.
- Only triggers a static safety warning when potentially urgent
  language is detected.

The normal RAG pipeline remains responsible for answering questions
from the uploaded leaflet.
"""

from __future__ import annotations

import re

from app.config import EMERGENCY_KEYWORDS


# ============================================================
# STATIC SAFETY MESSAGE
# ============================================================

EMERGENCY_MESSAGE = (
    "This question may describe a medical emergency. "
    "If you or someone else is experiencing a severe reaction, an overdose, "
    "difficulty breathing, chest pain, heavy bleeding, or loss of consciousness, "
    "contact emergency services or go to the nearest emergency room immediately. "
    "This assistant cannot provide emergency medical advice beyond what is written "
    "in the uploaded leaflet."
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def _normalize_text(text: str) -> str:
    """
    Normalize user text before keyword matching.

    This improves matching across:
    - uppercase/lowercase text
    - repeated whitespace
    - punctuation differences
    """

    if not text:
        return ""

    normalized = text.lower().strip()

    # Normalize repeated whitespace.
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized


# ============================================================
# KEYWORD MATCHING
# ============================================================

def detect_emergency(text: str) -> list[str]:
    """
    Return emergency keywords/phrases detected in the user's question.

    Matching is:
    - case-insensitive
    - whitespace-normalized
    - whole-word/whole-phrase based

    The returned list preserves the configured keyword values.
    """

    normalized_text = _normalize_text(text)

    if not normalized_text:
        return []

    matches: list[str] = []

    for keyword in EMERGENCY_KEYWORDS:
        normalized_keyword = _normalize_text(keyword)

        if not normalized_keyword:
            continue

        # Word boundaries prevent accidental partial matches.
        pattern = r"(?<!\w)" + re.escape(normalized_keyword) + r"(?!\w)"

        if re.search(pattern, normalized_text, flags=re.IGNORECASE):
            matches.append(keyword)

    return matches


# ============================================================
# BOOLEAN CHECK
# ============================================================

def is_emergency(text: str) -> bool:
    """
    Return True when potentially urgent language is detected.
    """

    return bool(detect_emergency(text))


# ============================================================
# SAFETY MESSAGE
# ============================================================

def get_emergency_message(text: str) -> str | None:
    """
    Return the static emergency warning when urgent language is detected.

    Returns:
        Warning message if emergency keywords are detected.
        None otherwise.
    """

    if is_emergency(text):
        return EMERGENCY_MESSAGE

    return None