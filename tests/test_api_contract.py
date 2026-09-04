"""
test_api_contract.py — the HTTP contract the dashboard + Mattermost share
(Sections 22, 33, 38). Verifies /health, /ready, /rag/status and /api/query
without needing a built index (the RAG call is stubbed) so it runs in CI.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import integrations.mattermost_bot as bot  # noqa: E402
from src import rag_service  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(bot.app)


def test_health_is_fast_and_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_rag_status_exposes_safe_operational_fields_only(client):
    s = client.get("/rag/status").json()
    for key in ("kb_version", "llm_model", "embedding_model", "available_models"):
        assert key in s
    assert s["llm_model"] == "openai/gpt-oss-120b"
    assert s["available_models"] == ["openai/gpt-oss-120b"]
    # Never leak secrets through status.
    dump = str(s).lower()
    assert "gsk_" not in dump and "password" not in dump and "api_key" not in dump


def test_api_query_returns_contract(client, monkeypatch):
    canned = {
        "request_id": "abc123", "query": "What is ATP?",
        "answer": "All Things Policy (ATP) is Takshashila's podcast. [Source 1]",
        "grounded": True, "confidence": "HIGH",
        "sources": [{"title": "All Things Policy", "url": "https://takshashila.org.in/atp/",
                     "source_type": "website", "snippet": "…"}],
        "kb_version": "20260101T000000Z-deadbeef", "model": "openai/gpt-oss-120b",
        "interface": "dashboard", "latency_ms": 42, "retrieval_ms": 5, "generation_ms": 30,
    }
    monkeypatch.setattr(rag_service, "answer_query", lambda *a, **k: canned)
    r = client.post("/api/query", json={"query": "What is ATP?", "interface": "dashboard"})
    assert r.status_code == 200
    body = r.json()
    for key in ("answer", "grounded", "confidence", "sources", "kb_version", "model", "latency_ms"):
        assert key in body
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["sources"][0]["url"].startswith("https://takshashila.org.in/")


def test_api_query_rejects_empty_and_oversized(client):
    assert client.post("/api/query", json={"query": ""}).status_code == 400
    assert client.post("/api/query", json={"query": "x" * 2001}).status_code == 413


def test_api_key_enforced_when_set(client, monkeypatch):
    monkeypatch.setattr(bot, "API_KEY", "s3cret")
    monkeypatch.setattr(rag_service, "answer_query", lambda *a, **k: {"answer": "ok"})
    assert client.post("/api/query", json={"query": "hi"}).status_code == 401
    ok = client.post("/api/query", json={"query": "hi"}, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200
