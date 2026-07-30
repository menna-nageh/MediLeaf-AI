"""
emergency.py
------------
Lightweight, deterministic keyword screen for potentially urgent
situations mentioned in the user's *question* (e.g. "I think I took too
much, what do I do?").

This is intentionally a simple keyword match rather than another LLM call:
it must be fast, fully deterministic, and never itself provide medical
advice - it only decides whether to surface a static warning banner
telling the user to seek urgent care. The actual answer text still comes
only from the leaflet, via the normal RAG pipeline.
"""

from __future__ import annotations

import re

from app.config import EMERGENCY_KEYWORDS

EMERGENCY_MESSAGE = (
    "This question may describe a medical emergency. "
    "If you or someone else is experiencing a severe reaction, an overdose, "
    "difficulty breathing, chest pain, heavy bleeding, or loss of consciousness, "
    "contact emergency services or go to the nearest emergency room immediately. "
    "This assistant cannot provide emergency medical advice beyond what is written "
    "in the uploaded leaflet."
)


def detect_emergency(text: str) -> list[str]:
    """
    Return the list of emergency keywords found in `text` (case-insensitive,
    whole-phrase matching). An empty list means no emergency terms were
    detected.
    """
    lowered = text.lower()
    matches = []
    for keyword in EMERGENCY_KEYWORDS:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, lowered):
            matches.append(keyword)
    return matches


def is_emergency(text: str) -> bool:
    return bool(detect_emergency(text))
