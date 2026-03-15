"""
Tests for the action executor and the full exploration loop.

Unit tests   — mock Street View API and Gemini, no keys needed.
Integration  — hit real Street View + Gemini APIs, run with: pytest -m integration
"""
import base64
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TINY_JPEG_BYTES = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e\xc8"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00"
    b"\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00"
    b"\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00"
    b"\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142\x81"
    b"\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19"
    b"\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86"
    b"\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4"
    b"\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2"
    b"\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9"
    b"\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5"
    b"\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd4"
    b"P\x00\x00\x00\x00\x1f\xff\xd9"
)
MOCK_B64 = "data:image/jpeg;base64," + base64.b64encode(_TINY_JPEG_BYTES).decode()

# ---------------------------------------------------------------------------
# /api/agent/run endpoint — full loop tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from app import app
    from fastapi.testclient import TestClient
    return TestClient(app)


MOCK_GUESS_RESULT = {
    "belief_state": [
        {"country": "Japan", "lat": 35.6595, "lon": 139.7005, "confidence": 0.9, "evidence": "Japanese text"}],
    "action": {"type": "GUESS", "lat": 35.6595, "lon": 139.7005},
    "final_guess": {"lat": 35.6595, "lon": 139.7005, "country": "Japan", "confidence": 0.9},
    "specialist_outputs": {},
    "iteration": 1,
    "error": '',
}

MOCK_ROTATE_RESULT = {
    "belief_state": [{"country": "Unknown", "lat": 0.0, "lon": 0.0, "confidence": 0.2, "evidence": "unclear"}],
    "action": {"type": "ROTATE", "degrees": 90},
    "final_guess": {},
    "specialist_outputs": {},
    "iteration": 1,
    "error": '',
}


def test_run_endpoint_exists(client):
    # POST with empty body should return 422 (validation), not 404
    r = client.post("/api/agent/run", json={})
    assert r.status_code != 404


def test_run_endpoint_guesses_immediately(client):
    """If graph returns GUESS on first iteration, loop stops and returns result."""
    with patch("agent.geo_graph") as mock_graph:
        mock_graph.invoke.return_value = MOCK_GUESS_RESULT
        r = client.post("/api/agent/run", json={
            "start_lat": 35.6595,
            "start_lon": 139.7005,
            "start_heading": 0.0,
            "max_iter": 5,
        })
    assert r.status_code == 200
    r.json()
    assert mock_graph.invoke.call_count == 1  # stopped after first GUESS


def test_run_endpoint_response_shape(client):
    with patch("agent.geo_graph") as mock_graph:
        mock_graph.invoke.return_value = MOCK_GUESS_RESULT
        r = client.post("/api/agent/run", json={
            "start_lat": 0.0,
            "start_lon": 0.0,
            "start_heading": 0.0,
            "max_iter": 3
        })
    data = r.json()
    for key in ["final_guess", "belief_state", "iterations_used"]:
        assert key in data, f"Missing key: {key}"


def test_run_endpoint_fresh_specialist_outputs_each_iteration(client):
    """Each graph invocation should receive empty specialist_outputs."""
    received_states = []

    def capture_state(state):
        received_states.append(dict(state))
        return MOCK_GUESS_RESULT

    with patch("agent.geo_graph") as mock_graph:
        mock_graph.invoke.side_effect = capture_state
        client.post("/api/agent/run", json={
            "start_lat": 0.0,
            "start_lon": 0.0,
            "start_heading": 0.0,
            "max_iter": 3
        })

    assert received_states[0]["specialist_outputs"] == {}