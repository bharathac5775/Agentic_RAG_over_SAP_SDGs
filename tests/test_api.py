"""Phase 6 FastAPI surface tests.

Uses FastAPI's TestClient — no real server, no real port. Calls the
handlers in-process. Pipeline is mocked so we test HTTP plumbing only.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.schemas import Citation, QueryResponse


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:

    def test_health_returns_valid_shape(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        # Required fields per HealthResponse schema
        assert "status" in body
        assert body["status"] in ("ok", "degraded")
        assert "ollama_reachable" in body
        assert "models_present" in body
        assert "index_present" in body
        assert "chunk_count" in body

    def test_health_reports_degraded_when_ollama_unreachable(self, client):
        with patch("app.api.ollama.Client") as mock_client:
            mock_client.return_value.list.side_effect = RuntimeError("connection refused")
            r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "degraded"
        assert body["ollama_reachable"] is False


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------


class TestQuery:

    def test_query_returns_valid_response(self, client):
        canned = QueryResponse(
            answer="An Active User is any individual who accesses the Cloud Service.",
            citations=[Citation(doc="X", section="1.2", page=1, quote="x")],
            verified=True,
            refused=False,
            warning=None,
            trace=None,
        )
        with patch("app.api.pipeline.answer", return_value=canned):
            r = client.post("/query", json={"question": "What is an Active User?"})

        assert r.status_code == 200
        body = r.json()
        assert "Active User" in body["answer"]
        assert body["verified"] is True
        assert body["refused"] is False
        assert len(body["citations"]) == 1
        assert body["citations"][0]["section"] == "1.2"

    def test_query_with_debug_returns_trace(self, client):
        canned = QueryResponse(
            answer="x", citations=[], verified=True, refused=False,
            trace={"latency_ms": {"total": 100}, "route": {"intent": "general"}},
        )
        with patch("app.api.pipeline.answer", return_value=canned) as mock_pipe:
            r = client.post("/query", json={"question": "anything", "debug": True})
            mock_pipe.assert_called_once()
            assert mock_pipe.call_args.kwargs["debug"] is True

        assert r.status_code == 200
        body = r.json()
        assert body["trace"] is not None
        assert body["trace"]["latency_ms"]["total"] == 100

    def test_query_refusal_returns_refused_true(self, client):
        canned = QueryResponse(
            answer="I won't write code.",
            citations=[],
            verified=True,
            refused=True,
            refusal_reason="matched code",
        )
        with patch("app.api.pipeline.answer", return_value=canned):
            r = client.post("/query", json={"question": "Write me a script"})

        body = r.json()
        assert body["refused"] is True
        assert body["refusal_reason"] == "matched code"

    def test_query_validates_input(self, client):
        # Missing question → 422 from FastAPI's Pydantic validation
        r = client.post("/query", json={})
        assert r.status_code == 422

        # Empty question → 422 because schema enforces min_length=1
        r = client.post("/query", json={"question": ""})
        assert r.status_code == 422

    def test_query_rejects_question_too_long(self, client):
        # Schema enforces max_length=2000
        too_long = "x" * 3000
        r = client.post("/query", json={"question": too_long})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# /docs / OpenAPI / root
# ---------------------------------------------------------------------------


class TestMeta:

    def test_root_returns_self_description(self, client):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert "service" in body
        assert "/docs" in body.get("docs", "")

    def test_openapi_schema_is_generated(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        # The two main endpoints must be present
        assert "/health" in schema["paths"]
        assert "/query" in schema["paths"]
        assert "POST" in {m.upper() for m in schema["paths"]["/query"].keys()}

    def test_swagger_ui_is_served(self, client):
        r = client.get("/docs")
        assert r.status_code == 200
        # Swagger UI is HTML; we only check the page returns content.
        assert "swagger" in r.text.lower() or "openapi" in r.text.lower()
