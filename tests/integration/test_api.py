"""Integration tests for FastAPI endpoints."""

import uuid
from typing import Any, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_pipeline_response() -> Dict[str, Any]:
    return {
        "answer": "Found 3 cardiologists in California.",
        "sources": [
            {
                "id": 1,
                "npi": "1234567890",
                "provider_name": "Dr. Jane Doe",
                "snippet": "Dr. Doe accepts Medicare.",
                "relevance_score": 0.92,
            }
        ],
        "suggestions": ["Show more in Southern California"],
        "providers_map": [
            {
                "npi": "1234567890",
                "lat": 34.05,
                "long": -118.24,
                "name": "Dr. Jane Doe",
                "specialty": "Cardiology",
            }
        ],
        "session_id": str(uuid.uuid4()),
        "latency_ms": 1200,
        "cache_hit": False,
        "model_used": "claude-sonnet-4-5",
    }


class TestHealthEndpoint:
    def test_health_returns_200(
        self, client: TestClient
    ) -> None:
        with (
            patch(
                "app.api.health.engine"
            ) as mock_engine,
            patch(
                "app.api.health.check_redis",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "app.api.health.check_embed_api",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            mock_conn = AsyncMock()
            mock_engine.connect.return_value.__aenter__ = (
                AsyncMock(return_value=mock_conn)
            )
            mock_engine.connect.return_value.__aexit__ = (
                AsyncMock(return_value=False)
            )
            mock_conn.execute = AsyncMock()
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(
        self, client: TestClient
    ) -> None:
        with (
            patch("app.api.health.engine") as mock_engine,
            patch(
                "app.api.health.check_redis",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "app.api.health.check_embed_api",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            mock_conn = AsyncMock()
            mock_engine.connect.return_value.__aenter__ = (
                AsyncMock(return_value=mock_conn)
            )
            mock_engine.connect.return_value.__aexit__ = (
                AsyncMock(return_value=False)
            )
            mock_conn.execute = AsyncMock()
            resp = client.get("/health")

        data = resp.json()
        assert "status" in data
        assert "db" in data
        assert "redis" in data
        assert "version" in data
        assert "uptime_seconds" in data


class TestQueryEndpoint:
    def test_query_returns_200(
        self,
        client: TestClient,
        mock_pipeline_response: Dict[str, Any],
    ) -> None:
        from app.schemas import QueryResponse

        mock_resp = QueryResponse(**mock_pipeline_response)
        with patch(
            "app.api.query.process_query",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            resp = client.post(
                "/query",
                json={"query": "cardiologists in California"},
            )
        assert resp.status_code == 200

    def test_query_response_has_answer(
        self,
        client: TestClient,
        mock_pipeline_response: Dict[str, Any],
    ) -> None:
        from app.schemas import QueryResponse

        mock_resp = QueryResponse(**mock_pipeline_response)
        with patch(
            "app.api.query.process_query",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            resp = client.post(
                "/query",
                json={"query": "cardiologists in California"},
            )
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert "latency_ms" in data
        assert "cache_hit" in data

    def test_query_too_short_returns_422(
        self, client: TestClient
    ) -> None:
        resp = client.post("/query", json={"query": "a"})
        assert resp.status_code == 422

    def test_query_too_long_returns_422(
        self, client: TestClient
    ) -> None:
        resp = client.post(
            "/query", json={"query": "x" * 2001}
        )
        assert resp.status_code == 422

    def test_missing_query_field_returns_422(
        self, client: TestClient
    ) -> None:
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_killswitch_returns_503(
        self, client: TestClient
    ) -> None:
        with patch(
            "app.api.query.settings"
        ) as mock_settings:
            mock_settings.query_killswitch = True
            resp = client.post(
                "/query",
                json={"query": "cardiologists in CA"},
            )
        assert resp.status_code == 503


class TestSessionEndpoint:
    def test_get_session_returns_200(
        self, client: TestClient
    ) -> None:
        session_id = str(uuid.uuid4())
        with patch(
            "app.api.session.get_session_history",
            new_callable=AsyncMock,
            return_value=[],
        ):
            resp = client.get(f"/session/{session_id}")
        assert resp.status_code == 200

    def test_delete_session_returns_204(
        self, client: TestClient
    ) -> None:
        session_id = str(uuid.uuid4())
        with patch(
            "app.api.session.clear_session",
            new_callable=AsyncMock,
        ):
            resp = client.delete(f"/session/{session_id}")
        assert resp.status_code == 204

    def test_invalid_uuid_returns_422(
        self, client: TestClient
    ) -> None:
        resp = client.get("/session/not-a-uuid")
        assert resp.status_code == 422


class TestIngestEndpoint:
    def test_status_returns_200(
        self, client: TestClient
    ) -> None:
        with patch(
            "app.api.ingest.get_latest_ingest_job",
            new_callable=AsyncMock,
            return_value=None,
        ):
            resp = client.get("/ingest/status")
        assert resp.status_code == 200

    def test_trigger_requires_admin_key(
        self, client: TestClient
    ) -> None:
        resp = client.post("/ingest/trigger")
        assert resp.status_code in (403, 422)

    def test_trigger_with_wrong_key_returns_403(
        self, client: TestClient
    ) -> None:
        with patch(
            "app.api.ingest.validate_admin_key",
            return_value=False,
        ):
            resp = client.post(
                "/ingest/trigger",
                headers={"X-Admin-Key": "wrong_key"},
            )
        assert resp.status_code == 403
