import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import serve as serve_module
from fastapi.testclient import TestClient
from serve import app

client = TestClient(app)

SIMPLE_REQ = {
    "nodes": ["A", "B", "C"],
    "edges": [
        {"from": "A", "to": "B", "snr": 8.0},
        {"from": "B", "to": "C", "snr": 7.0},
    ],
    "source": "A",
    "target": "C",
}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)


# ---------------------------------------------------------------------------
# /route — validation (no model needed)
# ---------------------------------------------------------------------------

def test_route_same_source_target(monkeypatch):
    monkeypatch.setattr(serve_module, "_model", None)
    req  = {**SIMPLE_REQ, "source": "A", "target": "A"}
    resp = client.post("/route", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["path"] == ["A"]
    assert data["score"] == 1.0


def test_route_unknown_source(monkeypatch):
    monkeypatch.setattr(serve_module, "_model", None)
    req  = {**SIMPLE_REQ, "source": "Z"}
    resp = client.post("/route", json=req)
    assert resp.status_code == 400


def test_route_unknown_target(monkeypatch):
    monkeypatch.setattr(serve_module, "_model", None)
    req  = {**SIMPLE_REQ, "target": "Z"}
    resp = client.post("/route", json=req)
    assert resp.status_code == 400


def test_route_no_model(monkeypatch):
    monkeypatch.setattr(serve_module, "_model", None)
    resp = client.post("/route", json=SIMPLE_REQ)
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /route — with a mock model
# ---------------------------------------------------------------------------

class _MockModel:
    """Always routes source → target directly (2-node path)."""

    def route(self, nf, ei, ew, src_i, dst_i, max_hops=20):
        return [src_i, dst_i]


def test_route_with_mock_model(monkeypatch):
    monkeypatch.setattr(serve_module, "_model", _MockModel())
    resp = client.post("/route", json=SIMPLE_REQ)
    assert resp.status_code == 200
    data = resp.json()
    assert data["path"][0]  == "A"
    assert data["path"][-1] == "C"
    assert data["method"]   == "ml"
    assert 0 < data["score"] <= 1.0


def test_route_no_path_from_model(monkeypatch):
    class _NoneModel:
        def route(self, *a, **kw):
            return None

    monkeypatch.setattr(serve_module, "_model", _NoneModel())
    resp = client.post("/route", json=SIMPLE_REQ)
    assert resp.status_code == 404


def test_health_model_loaded(monkeypatch):
    monkeypatch.setattr(serve_module, "_model", _MockModel())
    resp = client.get("/health")
    assert resp.json()["model_loaded"] is True
