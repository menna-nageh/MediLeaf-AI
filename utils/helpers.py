"""
helpers.py
----------
Small, dependency-free helper functions shared across the app (session id
generation, byte-size formatting, etc.). Anything here should be pure and
easy to unit test in isolation.
"""

from __future__ import annotations

import uuid


def new_session_id() -> str:
    """Generate a fresh, unique id for a Streamlit session / uploaded leaflet."""
    return uuid.uuid4().hex


def human_file_size(num_bytes: int) -> str:
    """Render a byte count as a short human-readable string, e.g. '482.1 KB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def truncate(text: str, max_len: int = 120) -> str:
    """Truncate long strings for compact display (e.g. sidebar history items)."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"
