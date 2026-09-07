"""Lightweight SQLite monitoring for MediLeaf requests and feedback."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.config import LOGS_DIR


DATABASE_PATH = Path(
    str(LOGS_DIR / "medileaf_monitoring.db")
)


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                latency_seconds REAL NOT NULL,
                retrieval_mode TEXT NOT NULL,
                confidence REAL NOT NULL,
                grounding_score REAL NOT NULL,
                grounded INTEGER NOT NULL,
                insufficient_information INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                helpful INTEGER NOT NULL
            )
            """
        )


def record_request(payload: dict[str, Any]) -> None:
    initialize_database()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO requests (
                timestamp, session_id, question, answer,
                latency_seconds, retrieval_mode, confidence,
                grounding_score, grounded, insufficient_information
            ) VALUES (
                datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                payload.get("session_id", ""),
                payload.get("question", ""),
                payload.get("answer", ""),
                float(payload.get("latency_seconds", 0.0)),
                payload.get("retrieval_mode", ""),
                float(payload.get("confidence", 0.0)),
                float(payload.get("grounding_score", 0.0)),
                int(bool(payload.get("grounded", False))),
                int(bool(payload.get("insufficient_information", False))),
            ),
        )


def record_feedback(
    session_id: str,
    question: str,
    helpful: bool,
) -> None:
    initialize_database()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO feedback (
                timestamp, session_id, question, helpful
            ) VALUES (datetime('now'), ?, ?, ?)
            """,
            (session_id, question, int(helpful)),
        )