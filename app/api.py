"""FastAPI interface for the existing MediLeaf RAG pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.monitoring import initialize_database, record_feedback
from app.service import ask
from utils.logger import log_error, log_feedback


app = FastAPI(title=settings.app_name, version="1.0.0")
initialize_database()


class AskRequest(BaseModel):
    session_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    memory_context: str = ""
    retrieval_mode: str | None = None


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    helpful: bool


def _answer_payload(request: AskRequest) -> dict:
    result = ask(
        session_id=request.session_id,
        question=request.question,
        memory_context=request.memory_context,
        retrieval_mode=request.retrieval_mode,
    )
    return {
        "answer": asdict(result.answer),
        "retrieval_count": result.retrieval_count,
        "retrieval_mode": result.retrieval_mode,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.post("/ask")
def ask_endpoint(request: AskRequest) -> dict:
    try:
        return _answer_payload(request)
    except Exception as exc:
        log_error(request.session_id, "api.ask", str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask/stream")
def ask_stream_endpoint(request: AskRequest) -> StreamingResponse:
    def events() -> Iterator[str]:
        try:
            yield json.dumps(_answer_payload(request), default=str) + "\n"
        except Exception as exc:
            log_error(request.session_id, "api.ask.stream", str(exc))
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/feedback")
def feedback_endpoint(request: FeedbackRequest) -> dict[str, str]:
    log_feedback(request.session_id, request.question, request.helpful)
    record_feedback(request.session_id, request.question, request.helpful)
    return {"status": "recorded"}