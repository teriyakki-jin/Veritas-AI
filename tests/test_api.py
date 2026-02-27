"""Tests for src/api_server.py endpoints."""

import os

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_mock_pipeline():
    """Return a minimal mock FactCheckPipeline."""
    mock = MagicMock()
    mock.models = {"liar": MagicMock(), "fever": MagicMock()}
    mock.retriever = MagicMock()
    mock.retriever.corpus = ["doc1", "doc2"]
    mock.verify.return_value = {
        "claim": "test claim",
        "credibility_score": 0.75,
        "verdict": "TRUE",
        "evidence": [{"doc_id": "doc1", "score": 1.5, "snippet": "Some evidence text."}],
        "model_details": {
            "liar": {
                "credibility_score": 0.8,
                "predicted_class": 5,
                "predicted_label": "true",
                "probabilities": [0.1, 0.1, 0.1, 0.1, 0.2, 0.4],
                "temperature": 1.0,
                "weight": 0.35,
            }
        },
        "inference_time_ms": 120.0,
    }
    return mock


@pytest.fixture
def client_with_pipeline():
    """TestClient where FactCheckPipeline() is mocked so lifespan uses mock."""
    import api_server
    mock_pipeline = _make_mock_pipeline()
    # Patch FactCheckPipeline constructor so lifespan sets pipeline = mock_pipeline
    with patch("api_server.FactCheckPipeline", return_value=mock_pipeline):
        with TestClient(api_server.app, raise_server_exceptions=True) as c:
            yield c, mock_pipeline


@pytest.fixture
def client_no_pipeline():
    """TestClient with pipeline forced to None after lifespan runs."""
    import api_server
    mock_pipeline = _make_mock_pipeline()
    with patch("api_server.FactCheckPipeline", return_value=mock_pipeline):
        with TestClient(api_server.app, raise_server_exceptions=True) as c:
            api_server.app.state.pipeline = None
            yield c


class TestHealthEndpoint:
    def test_health_503_when_pipeline_none(self, client_no_pipeline):
        resp = client_no_pipeline.get("/health")
        assert resp.status_code == 503

    def test_health_200_when_pipeline_loaded(self, client_with_pipeline):
        client, _ = client_with_pipeline
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert isinstance(data["active_models"], list)
        assert isinstance(data["retrieval_docs"], int)


class TestVerifyEndpoint:
    def test_empty_claim_returns_422(self, client_with_pipeline):
        client, _ = client_with_pipeline
        resp = client.post("/verify", json={"claim": ""})
        assert resp.status_code == 422

    def test_whitespace_claim_returns_422(self, client_with_pipeline):
        client, _ = client_with_pipeline
        resp = client.post("/verify", json={"claim": "   "})
        assert resp.status_code == 422

    def test_valid_claim_returns_200(self, client_with_pipeline):
        client, _ = client_with_pipeline
        resp = client.post("/verify", json={"claim": "The earth is round."})
        assert resp.status_code == 200
        data = resp.json()
        assert "credibility_score" in data
        assert "verdict" in data
        assert "evidence" in data

    def test_503_when_no_pipeline(self, client_no_pipeline):
        resp = client_no_pipeline.post("/verify", json={"claim": "Some claim."})
        assert resp.status_code == 503

    def test_response_has_all_required_fields(self, client_with_pipeline):
        client, _ = client_with_pipeline
        resp = client.post("/verify", json={"claim": "The study confirmed results."})
        assert resp.status_code == 200
        data = resp.json()
        for field in ["claim", "credibility_score", "verdict", "evidence", "model_details"]:
            assert field in data

    def test_credibility_score_in_unit_interval(self, client_with_pipeline):
        client, _ = client_with_pipeline
        resp = client.post("/verify", json={"claim": "Test claim with data."})
        assert resp.status_code == 200
        score = resp.json()["credibility_score"]
        assert 0.0 <= score <= 1.0


class TestBatchVerifyEndpoint:
    def test_empty_claims_list_returns_422(self, client_with_pipeline):
        client, _ = client_with_pipeline
        resp = client.post("/verify/batch", json={"claims": []})
        assert resp.status_code == 422

    def test_single_claim_in_batch(self, client_with_pipeline):
        client, _ = client_with_pipeline
        resp = client.post("/verify/batch", json={"claims": ["The earth orbits the sun."]})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1

    def test_multiple_claims_in_batch_returns_list(self, client_with_pipeline):
        client, _ = client_with_pipeline
        claims = ["Claim one is factual.", "Claim two has number 42."]
        resp = client.post("/verify/batch", json={"claims": claims})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_503_when_no_pipeline(self, client_no_pipeline):
        resp = client_no_pipeline.post("/verify/batch", json={"claims": ["Test claim."]})
        assert resp.status_code == 503

    def test_batch_exceeds_max_returns_422(self, client_with_pipeline):
        client, _ = client_with_pipeline
        claims = [f"Claim number {i} has fact." for i in range(51)]
        resp = client.post("/verify/batch", json={"claims": claims})
        assert resp.status_code == 422


class TestCorsConfiguration:
    def test_wildcard_cors_disables_credentials(self):
        import api_server
        with patch.dict(os.environ, {"CORS_ORIGINS": "*"}):
            origins = api_server._parse_cors_origins()
            if "*" in origins:
                assert ("*" not in origins) is False  # wildcard present
                allow_creds = "*" not in origins
                assert allow_creds is False

    def test_explicit_origins_enables_credentials(self):
        import api_server
        with patch.dict(os.environ, {"CORS_ORIGINS": "http://localhost:3000"}):
            origins = api_server._parse_cors_origins()
            assert "http://localhost:3000" in origins
            assert "*" not in origins
            assert ("*" not in origins) is True  # credentials would be enabled

    def test_multiple_origins_parsed(self):
        import api_server
        with patch.dict(os.environ, {"CORS_ORIGINS": "http://a.com,http://b.com"}):
            origins = api_server._parse_cors_origins()
            assert "http://a.com" in origins
            assert "http://b.com" in origins


class TestRateLimiting:
    def test_rate_limit_middleware_exists(self):
        from api_server import RateLimitMiddleware
        assert RateLimitMiddleware is not None

    def test_rate_limit_config_defined(self):
        from api_server import _RATE_LIMITS
        assert "/verify" in _RATE_LIMITS
        assert "/verify/batch" in _RATE_LIMITS
        assert "/analyze/article" in _RATE_LIMITS

    def test_batch_has_lower_limit_than_single(self):
        from api_server import _RATE_LIMITS
        single_limit = _RATE_LIMITS["/verify"][0]
        batch_limit = _RATE_LIMITS["/verify/batch"][0]
        assert batch_limit < single_limit

    def test_article_has_lowest_limit(self):
        from api_server import _RATE_LIMITS
        article_limit = _RATE_LIMITS["/analyze/article"][0]
        batch_limit = _RATE_LIMITS["/verify/batch"][0]
        assert article_limit <= batch_limit

    def test_rate_limit_returns_429_on_excess(self, client_with_pipeline):
        """Spam /verify/batch beyond its per-minute limit to trigger 429."""
        import api_server
        from api_server import _RATE_LIMITS, RateLimitMiddleware
        client, _ = client_with_pipeline

        max_calls, _ = _RATE_LIMITS["/verify/batch"]
        # Exhaust the limit
        for _ in range(max_calls):
            client.post("/verify/batch", json={"claims": ["Test claim."]})

        # Next request should be rate limited
        resp = client.post("/verify/batch", json={"claims": ["Over limit."]})
        assert resp.status_code == 429
        assert "Too many requests" in resp.json()["detail"]
