"""
logger.py
---------
Structured JSON-lines logging for MediLeaf AI.

Every question asked, the retriever results returned for it, execution
time, and any errors are appended as one JSON object per line to
`logs/medileaf.log`. JSON-lines is used (rather than a single JSON array
or plain text) because it is append-safe, human-readable, and trivial to
load into pandas for later analysis.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from app.config import LOGS_DIR

LOG_FILE = LOGS_DIR / "medileaf.log"

# A standard Python logger is used underneath so log rotation / handlers can
# be extended later without changing call sites.
_logger = logging.getLogger("medileaf")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)


def _write(event_type: str, payload: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": event_type,
        **payload,
    }
    try:
        _logger.info(json.dumps(record, ensure_ascii=False, default=str))
    except Exception:
        # Logging must never crash the app - swallow any serialization issue.
        pass


def log_question(session_id: str, question: str) -> None:
    _write("question", {"session_id": session_id, "question": question})


def log_retrieval(
    session_id: str,
    question: str,
    num_results: int,
    top_confidence: float,
    execution_time_seconds: float,
    retrieval_mode: str = "",
) -> None:
    _write(
        "retrieval",
        {
            "session_id": session_id,
            "question": question,
            "num_results": num_results,
            "top_confidence": top_confidence,
            "execution_time_seconds": execution_time_seconds,
            "retrieval_mode": retrieval_mode,
        },
    )


def log_answer(
    session_id: str,
    question: str,
    execution_time_seconds: float,
    insufficient: bool,
    confidence: float = 0.0,
    grounded: bool = False,
    grounding_score: float = 0.0,
    retrieval_mode: str = "",
) -> None:
    _write(
        "answer",
        {
            "session_id": session_id,
            "question": question,
            "execution_time_seconds": execution_time_seconds,
            "insufficient_information": insufficient,
            "confidence": confidence,
            "grounded": grounded,
            "grounding_score": grounding_score,
            "retrieval_mode": retrieval_mode,
        },
    )


def log_error(session_id: str, stage: str, error_message: str) -> None:
    _write("error", {"session_id": session_id, "stage": stage, "error_message": error_message})


def log_feedback(session_id: str, question: str, helpful: bool) -> None:
    _write("feedback", {"session_id": session_id, "question": question, "helpful": helpful})
