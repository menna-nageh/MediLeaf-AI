"""Shared RAG orchestration used by Streamlit, API, and evaluations."""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.config import settings
from app.llm import StructuredAnswer, answer_question
from app.monitoring import record_request
from app.retriever import load_vector_store, overall_confidence, retrieve
from utils.logger import log_answer, log_question, log_retrieval


@dataclass
class AskResult:
    answer: StructuredAnswer
    retrieval_count: int
    retrieval_mode: str


def ask(
    session_id: str,
    question: str,
    memory_context: str = "",
    retrieval_mode: str | None = None,
) -> AskResult:
    """Run the existing retrieval and generation pipeline for a session."""

    if not session_id or not session_id.strip():
        raise ValueError("session_id cannot be empty")
    if not question or not question.strip():
        raise ValueError("question cannot be empty")

    started = time.perf_counter()
    mode = (retrieval_mode or settings.retrieval_mode).lower()
    log_question(session_id, question)

    store = load_vector_store(session_id)
    chunks = retrieve(
        store,
        question,
        mode=mode,
    )
    retrieval_latency = time.perf_counter() - started
    confidence = overall_confidence(chunks)
    log_retrieval(
        session_id,
        question,
        len(chunks),
        confidence,
        retrieval_latency,
        retrieval_mode=mode,
    )

    answer = answer_question(
        question,
        chunks,
        memory_context=memory_context,
        overall_confidence=confidence,
    )
    total_latency = time.perf_counter() - started
    log_answer(
        session_id,
        question,
        total_latency,
        answer.insufficient_information,
        confidence=answer.confidence,
        grounded=answer.grounded,
        grounding_score=answer.grounding_score,
        retrieval_mode=mode,
    )
    record_request(
        {
            "session_id": session_id,
            "question": question,
            "answer": answer.answer,
            "latency_seconds": total_latency,
            "retrieval_mode": mode,
            "confidence": answer.confidence,
            "grounded": answer.grounded,
            "grounding_score": answer.grounding_score,
            "insufficient_information": answer.insufficient_information,
        }
    )
    return AskResult(
        answer=answer,
        retrieval_count=len(chunks),
        retrieval_mode=mode,
    )