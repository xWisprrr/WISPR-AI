"""
Tests for the FastAPI application (using TestClient — no real LLM calls).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_contains_app_name(self, client):
        data = resp = client.get("/").json()
        assert "WISPR" in data["name"]

    def test_root_has_capabilities(self, client):
        data = client.get("/").json()
        assert "capabilities" in data
        assert len(data["capabilities"]) > 0

    def test_root_has_endpoints(self, client):
        data = client.get("/").json()
        assert "endpoints" in data
        assert "/query" in str(data["endpoints"])


class TestAgentsEndpoint:
    def test_agents_returns_200(self, client):
        resp = client.get("/agents")
        assert resp.status_code == 200

    def test_agents_has_builtin(self, client):
        data = client.get("/agents").json()
        names = [a["name"] for a in data["built_in_agents"]]
        assert "core" in names
        assert "coder" in names
        assert "search" in names
        assert "studio" in names
        assert "orchestrator" in names

    def test_agents_total_count(self, client):
        data = client.get("/agents").json()
        assert data["total"] >= 5


class TestPluginsEndpoint:
    def test_plugins_returns_200(self, client):
        resp = client.get("/plugins")
        assert resp.status_code == 200

    def test_plugins_has_count(self, client):
        data = client.get("/plugins").json()
        assert "count" in data


class TestMemoryEndpoint:
    def test_memory_get_returns_200(self, client):
        resp = client.get("/memory")
        assert resp.status_code == 200

    def test_memory_store_and_retrieve(self, client):
        payload = {"key": "test_key", "value": "test_value", "tags": ["test"]}
        resp = client.post("/memory", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stored"
        assert data["key"] == "test_key"


class TestQueryEndpoint:
    def test_query_missing_prompt(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422  # validation error

    @patch("api.routes._orchestrator._timed_run", new_callable=AsyncMock)
    def test_query_auto_mode(self, mock_run, client):
        from agents.base_agent import AgentResult

        mock_run.return_value = AgentResult(
            agent_id="test-id",
            agent_name="orchestrator",
            success=True,
            output="Test answer",
            metadata={"confidence": 0.9},
        )
        resp = client.post("/query", json={"prompt": "Hello", "mode": "auto"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Test answer"
        assert data["agent"] == "orchestrator"


class TestSearchEndpoint:
    @patch("api.routes._mega.search", new_callable=AsyncMock)
    def test_search_returns_results(self, mock_search, client):
        mock_search.return_value = [
            {
                "title": "Test",
                "url": "https://example.com",
                "snippet": "Test snippet",
                "source": "duckduckgo",
                "score": 1.0,
            }
        ]
        resp = client.post("/search", json={"query": "test query"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["result_count"] == 1
        assert len(data["results"]) == 1


class TestCodeEndpoint:
    @patch("api.routes._coder._timed_run", new_callable=AsyncMock)
    def test_code_generate(self, mock_run, client):
        from agents.base_agent import AgentResult

        mock_run.return_value = AgentResult(
            agent_id="test-id",
            agent_name="coder",
            success=True,
            output="```python\nprint('hello')\n```",
            metadata={
                "language": "python",
                "action": "generate",
                "code_blocks": [{"language": "python", "code": "print('hello')"}],
            },
        )
        resp = client.post(
            "/code",
            json={"task": "print hello world in python", "action": "generate"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["language"] == "python"
        assert len(data["code_blocks"]) == 1
