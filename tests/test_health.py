"""Tests for health endpoint."""

from __future__ import annotations


def test_health_model_loaded(client):
    """GET /health returns 200 with status=ok, workers_healthy, workers_total."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "workers_healthy" in data
    assert "workers_total" in data


def test_health_model_not_loaded(client_not_loaded):
    """GET /health returns 503 with status=starting when model is loading."""
    resp = client_not_loaded.get("/health")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "starting"
    assert data["workers_healthy"] == 0
    assert "workers_total" in data
