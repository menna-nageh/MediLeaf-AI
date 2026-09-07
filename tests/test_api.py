"""Contract tests for the FastAPI endpoints without calling Gemini."""

from dataclasses import asdict

from fastapi.testclient import TestClient

from app import api
from app.llm import StructuredAnswer
from app.service import AskResult


def _fake_ask(**kwargs):
    return AskResult(
        answer=StructuredAnswer(
            insufficient_information=False,
            answer="Supported answer",
            important_information="",
            warnings="",
            practical_advice="",
            explanation="",
            confidence=90.0,
        ),
        retrieval_count=1,
        retrieval_mode=kwargs.get("retrieval_mode") or "hybrid_rerank",
    )


def test_health_endpoint():
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_endpoint_uses_shared_service(monkeypatch):
    monkeypatch.setattr(api, "ask", _fake_ask)
    response = TestClient(api.app).post(
        "/ask",
        json={"session_id": "session-1", "question": "What is it for?"},
    )
    assert response.status_code == 200
    assert response.json()["answer"]["answer"] == "Supported answer"


def test_feedback_endpoint_records_request(monkeypatch):
    recorded = []
    monkeypatch.setattr(api, "record_feedback", lambda *args: recorded.append(args))
    response = TestClient(api.app).post(
        "/feedback",
        json={"session_id": "session-1", "question": "q", "helpful": True},
    )
    assert response.status_code == 200
    assert recorded == [("session-1", "q", True)]