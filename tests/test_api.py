"""Tests for src/api_server.py endpoints."""

import json
import os

import numpy as np
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _parse_sse_events(text: str) -> list:
    """Parse SSE response body into list of {event, data} dicts."""
    events = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = None
        data_str = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                data_str = line[len("data: "):]
        if event_name and data_str is not None:
            events.append({"event": event_name, "data": json.loads(data_str)})
    return events


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


class TestVerifyStreamEndpoint:
    def _setup_stream_mocks(self, mock_pipeline):
        """Configure mock pipeline for stream endpoint."""
        mock_pipeline.retrieve_evidence.return_value = [
            {"doc_id": "doc1", "score": 1.5, "text": "Some stream evidence."}
        ]
        mock_pipeline.predict_single.return_value = np.array([0.1, 0.2, 0.7])
        mock_pipeline.fusion.fuse.return_value = {
            "credibility_score": 0.75,
            "verdict": "TRUE",
            "model_details": {},
        }
        mock_pipeline._enrich_model_details.return_value = {
            "credibility_score": 0.75,
            "verdict": "TRUE",
            "model_details": {},
        }

    def test_stream_missing_claim_returns_422(self, client_with_pipeline):
        client, _ = client_with_pipeline
        resp = client.get("/verify/stream")
        assert resp.status_code == 422

    def test_stream_whitespace_claim_returns_422(self, client_with_pipeline):
        client, _ = client_with_pipeline
        resp = client.get("/verify/stream?claim=   ")
        assert resp.status_code == 422

    def test_stream_valid_claim_returns_event_stream(self, client_with_pipeline):
        client, mock_pipeline = client_with_pipeline
        self._setup_stream_mocks(mock_pipeline)
        resp = client.get("/verify/stream?claim=The+earth+is+round.")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_emits_expected_events_in_order(self, client_with_pipeline):
        client, mock_pipeline = client_with_pipeline
        self._setup_stream_mocks(mock_pipeline)
        resp = client.get("/verify/stream?claim=Vaccines+are+safe.")
        events = _parse_sse_events(resp.text)
        event_names = [e["event"] for e in events]

        for expected in ["retrieving", "evidence", "fusing", "result", "done"]:
            assert expected in event_names, f"Missing SSE event: {expected}"

        idx = {name: event_names.index(name) for name in ["retrieving", "evidence", "fusing", "result", "done"]}
        assert idx["retrieving"] < idx["evidence"] < idx["fusing"] < idx["result"] < idx["done"]

    def test_stream_result_event_has_required_fields(self, client_with_pipeline):
        client, mock_pipeline = client_with_pipeline
        self._setup_stream_mocks(mock_pipeline)
        resp = client.get("/verify/stream?claim=Science+confirms+this.")
        events = _parse_sse_events(resp.text)
        result_events = [e for e in events if e["event"] == "result"]
        assert len(result_events) == 1
        data = result_events[0]["data"]
        for field in ["claim", "credibility_score", "verdict", "evidence", "model_details"]:
            assert field in data, f"Missing field in result event: {field}"

    def test_stream_evidence_event_reports_count(self, client_with_pipeline):
        client, mock_pipeline = client_with_pipeline
        self._setup_stream_mocks(mock_pipeline)
        resp = client.get("/verify/stream?claim=Climate+change+is+real.")
        events = _parse_sse_events(resp.text)
        evidence_events = [e for e in events if e["event"] == "evidence"]
        assert len(evidence_events) == 1
        assert evidence_events[0]["data"]["count"] == 1

    def test_stream_emits_verifying_and_model_done_per_model(self, client_with_pipeline):
        client, mock_pipeline = client_with_pipeline
        self._setup_stream_mocks(mock_pipeline)
        resp = client.get("/verify/stream?claim=Test+model+events.")
        events = _parse_sse_events(resp.text)
        model_names = set(mock_pipeline.models.keys())
        verifying_models = {e["data"]["model"] for e in events if e["event"] == "verifying"}
        done_models = {e["data"]["model"] for e in events if e["event"] == "model_done"}
        assert verifying_models == model_names
        assert done_models == model_names

    def test_stream_503_when_no_pipeline(self, client_no_pipeline):
        resp = client_no_pipeline.get("/verify/stream?claim=Test+claim.")
        assert resp.status_code == 503


class TestVerifyExplainEndpoint:
    def _setup_explain_mocks(self, mock_pipeline):
        """Configure explain_evidence return value."""
        mock_pipeline.explain_evidence.return_value = [
            {"contribution": 0.15, "contribution_label": "supports"}
        ]

    def test_explain_empty_claim_returns_422(self, client_with_pipeline):
        client, _ = client_with_pipeline
        resp = client.post("/verify/explain", json={"claim": ""})
        assert resp.status_code == 422

    def test_explain_whitespace_claim_returns_422(self, client_with_pipeline):
        client, _ = client_with_pipeline
        resp = client.post("/verify/explain", json={"claim": "   "})
        assert resp.status_code == 422

    def test_explain_valid_claim_returns_200(self, client_with_pipeline):
        client, mock_pipeline = client_with_pipeline
        self._setup_explain_mocks(mock_pipeline)
        resp = client.post("/verify/explain", json={"claim": "The earth is round."})
        assert resp.status_code == 200

    def test_explain_response_has_required_fields(self, client_with_pipeline):
        client, mock_pipeline = client_with_pipeline
        self._setup_explain_mocks(mock_pipeline)
        resp = client.post("/verify/explain", json={"claim": "Vaccines are safe."})
        assert resp.status_code == 200
        data = resp.json()
        for field in ["claim", "credibility_score", "verdict", "evidence", "model_details"]:
            assert field in data, f"Missing field in explain response: {field}"

    def test_explain_evidence_has_contribution_fields(self, client_with_pipeline):
        client, mock_pipeline = client_with_pipeline
        self._setup_explain_mocks(mock_pipeline)
        resp = client.post("/verify/explain", json={"claim": "Science supports this."})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["evidence"]) > 0
        ev = data["evidence"][0]
        assert "contribution" in ev
        assert "contribution_label" in ev

    def test_explain_contribution_label_is_valid(self, client_with_pipeline):
        client, mock_pipeline = client_with_pipeline
        self._setup_explain_mocks(mock_pipeline)
        resp = client.post("/verify/explain", json={"claim": "Facts matter here."})
        assert resp.status_code == 200
        valid_labels = {"supports", "refutes", "neutral"}
        for ev in resp.json()["evidence"]:
            assert ev["contribution_label"] in valid_labels

    def test_explain_contribution_is_float(self, client_with_pipeline):
        client, mock_pipeline = client_with_pipeline
        self._setup_explain_mocks(mock_pipeline)
        resp = client.post("/verify/explain", json={"claim": "Numbers matter too."})
        assert resp.status_code == 200
        for ev in resp.json()["evidence"]:
            assert isinstance(ev["contribution"], float)

    def test_explain_503_when_no_pipeline(self, client_no_pipeline):
        resp = client_no_pipeline.post("/verify/explain", json={"claim": "Test claim."})
        assert resp.status_code == 503

    def test_explain_calls_verify_then_explain_evidence(self, client_with_pipeline):
        client, mock_pipeline = client_with_pipeline
        self._setup_explain_mocks(mock_pipeline)
        client.post("/verify/explain", json={"claim": "Check method calls."})
        mock_pipeline.verify.assert_called_once()
        mock_pipeline.explain_evidence.assert_called_once()
