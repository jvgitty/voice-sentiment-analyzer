"""Integration tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from vsa.api import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestAnalyzeEndpoint:
    def test_empty_body_returns_422(self, client: TestClient) -> None:
        response = client.post("/analyze", json={})
        assert response.status_code == 422
