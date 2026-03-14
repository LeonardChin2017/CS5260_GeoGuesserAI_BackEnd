"""
Tests for the action executor and the full exploration loop.

Unit tests   — mock Street View API and Gemini, no keys needed.
Integration  — hit real Street View + Gemini APIs, run with: pytest -m integration
"""
import base64
import json
import math
import os
from unittest.mock import patch, MagicMock

import pytest
from graphs.action_executor import GameView

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
MOCK_MAPS_KEY = "fake-maps-key"


# ---------------------------------------------------------------------------
# GameView unit tests
# ---------------------------------------------------------------------------

class TestGameView:
    def test_rotate_adds_degrees(self):
        from graphs.action_executor import GameView
        view = GameView(35.0, 139.0, heading=90)
        new = view.rotate(90)
        assert new.heading == 180
        assert new.lat == 35.0
        assert new.lon == 139.0

    def test_rotate_wraps_at_360(self):
        from graphs.action_executor import GameView
        view = GameView(35.0, 139.0, heading=300)
        new = view.rotate(90)
        assert new.heading == 30

    def test_rotate_negative_degrees(self):
        from graphs.action_executor import GameView
        view = GameView(35.0, 139.0, heading=10)
        new = view.rotate(-45)
        assert new.heading == 325

    def test_move_forward_changes_position(self):
        from graphs.action_executor import GameView
        view = GameView(35.0, 139.0, heading=0)
        new = view.move_forward()
        assert new.lat != view.lat or new.lon != view.lon

    def test_move_north_increases_lat(self):
        """Heading 0 = North → latitude increases."""
        from graphs.action_executor import GameView
        view = GameView(35.0, 139.0, heading=0)
        new = view.move_forward(metres=50)
        assert new.lat > view.lat
        assert abs(new.lon - view.lon) < 1e-6  # lon unchanged moving north

    def test_move_east_increases_lon(self):
        """Heading 90 = East → longitude increases."""
        from graphs.action_executor import GameView
        view = GameView(35.0, 139.0, heading=90)
        new = view.move_forward(metres=50)
        assert new.lon > view.lon
        assert abs(new.lat - view.lat) < 1e-6

    def test_move_preserves_heading(self):
        from graphs.action_executor import GameView
        view = GameView(35.0, 139.0, heading=45)
        new = view.move_forward()
        assert new.heading == 45

    def test_to_dict_and_from_dict(self):
        from graphs.action_executor import GameView
        view = GameView(35.6, 139.7, heading=120)
        d = view.to_dict()
        assert d == {"lat": 35.6, "lon": 139.7, "heading": 120}
        restored = GameView.from_dict(d)
        assert restored.lat == 35.6
        assert restored.lon == 139.7
        assert restored.heading == 120

    def test_move_distance_is_approximately_correct(self):
        """50m north should shift lat by ~0.00045 degrees."""
        from graphs.action_executor import GameView
        view = GameView(0.0, 0.0, heading=0)
        new = view.move_forward(metres=50)
        expected_delta = 50 / 111_000
        assert abs(new.lat - expected_delta) < 1e-7


# ---------------------------------------------------------------------------
# fetch_streetview_screenshot unit tests
# ---------------------------------------------------------------------------

class TestFetchStreetview:
    def test_raises_without_api_key(self):
        from graphs.action_executor import GameView, fetch_streetview_screenshot
        with pytest.raises(ValueError, match="GOOGLE_MAPS_API_KEY"):
            fetch_streetview_screenshot(GameView(0, 0), api_key="")

    def test_returns_data_url(self):
        from graphs.action_executor import GameView, fetch_streetview_screenshot
        mock_resp = MagicMock()
        mock_resp.content = _TINY_JPEG_BYTES
        mock_resp.raise_for_status = MagicMock()
        with patch("graphs.action_executor.requests.get", return_value=mock_resp):
            result = fetch_streetview_screenshot(GameView(35.0, 139.0, 0), MOCK_MAPS_KEY)
        assert result.startswith("data:image/jpeg;base64,")

    def test_correct_api_params_sent(self):
        from graphs.action_executor import GameView, fetch_streetview_screenshot
        mock_resp = MagicMock()
        mock_resp.content = _TINY_JPEG_BYTES
        mock_resp.raise_for_status = MagicMock()
        with patch("graphs.action_executor.requests.get", return_value=mock_resp) as mock_get:
            fetch_streetview_screenshot(GameView(35.6, 139.7, 90), MOCK_MAPS_KEY)
        params = mock_get.call_args[1]["params"]
        assert params["heading"] == 90
        assert "35.6" in params["location"]
        assert "139.7" in params["location"]
        assert params["key"] == MOCK_MAPS_KEY


# ---------------------------------------------------------------------------
# execute_action unit tests
# ---------------------------------------------------------------------------

class TestExecuteAction:
    def _mock_fetch(self):
        mock_resp = MagicMock()
        mock_resp.content = _TINY_JPEG_BYTES
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_rotate_updates_heading(self):
        from graphs.action_executor import GameView, execute_action
        view = GameView(35.0, 139.0, heading=0)
        with patch("graphs.action_executor.requests.get", return_value=self._mock_fetch()):
            _, new_view = execute_action({"type": "ROTATE", "degrees": 90}, view, MOCK_MAPS_KEY)
        assert new_view.heading == 90

    def test_move_updates_position(self):
        from graphs.action_executor import GameView, execute_action
        view = GameView(35.0, 139.0, heading=0)
        with patch("graphs.action_executor.requests.get", return_value=self._mock_fetch()):
            _, new_view = execute_action({"type": "MOVE", "direction": "forward"}, view, MOCK_MAPS_KEY)
        assert new_view.lat != view.lat

    def test_returns_new_screenshot(self):
        from graphs.action_executor import GameView, execute_action
        view = GameView(35.0, 139.0, heading=0)
        with patch("graphs.action_executor.requests.get", return_value=self._mock_fetch()):
            screenshot, _ = execute_action({"type": "ROTATE", "degrees": 45}, view, MOCK_MAPS_KEY)
        assert screenshot.startswith("data:image/jpeg;base64,")

    def test_raises_on_guess_action(self):
        from graphs.action_executor import GameView, execute_action
        view = GameView(35.0, 139.0)
        with pytest.raises(ValueError, match="GUESS"):
            execute_action({"type": "GUESS", "lat": 35.0, "lon": 139.0}, view, MOCK_MAPS_KEY)


# ---------------------------------------------------------------------------
# /api/agent/run endpoint — full loop tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from app import app
    from fastapi.testclient import TestClient
    return TestClient(app)


MOCK_GUESS_RESULT = {
    "belief_state": [{"country": "Japan", "lat": 35.6595, "lon": 139.7005, "confidence": 0.9, "evidence": "Japanese text"}],
    "action": {"type": "GUESS", "lat": 35.6595, "lon": 139.7005},
    "final_guess": {"lat": 35.6595, "lon": 139.7005, "country": "Japan", "confidence": 0.9},
    "specialist_outputs": {},
    "iteration": 1,
    "error": None,
}

MOCK_ROTATE_RESULT = {
    "belief_state": [{"country": "Unknown", "lat": 0.0, "lon": 0.0, "confidence": 0.2, "evidence": "unclear"}],
    "action": {"type": "ROTATE", "degrees": 90},
    "final_guess": None,
    "specialist_outputs": {},
    "iteration": 1,
    "error": None,
}


def test_run_endpoint_exists(client):
    # POST with empty body should return 422 (validation), not 404
    r = client.post("/api/agent/run", json={})
    assert r.status_code != 404


def test_run_endpoint_guesses_immediately(client):
    """If graph returns GUESS on first iteration, loop stops and returns result."""
    with patch("app.geo_graph") as mock_graph:
        mock_graph.invoke.return_value = MOCK_GUESS_RESULT
        r = client.post("/api/agent/run", json={
            "screenshot": MOCK_B64,
            "start_lat": 35.6595,
            "start_lon": 139.7005,
            "max_iterations": 5,
        })
    assert r.status_code == 200
    data = r.json()
    assert data["final_guess"]["lat"] == 35.6595
    assert mock_graph.invoke.call_count == 1  # stopped after first GUESS


def test_run_endpoint_loops_on_rotate_then_guesses(client):
    """Graph returns ROTATE first, then GUESS — should call graph twice."""
    mock_resp = MagicMock()
    mock_resp.content = _TINY_JPEG_BYTES
    mock_resp.raise_for_status = MagicMock()

    call_count = {"n": 0}
    def side_effect(state):
        call_count["n"] += 1
        return MOCK_GUESS_RESULT if call_count["n"] > 1 else MOCK_ROTATE_RESULT

    with patch("app.geo_graph") as mock_graph, \
         patch("app.execute_action", return_value=(MOCK_B64, GameView(35.0, 139.0, 90))) as mock_exec, \
         patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": MOCK_MAPS_KEY}):
        mock_graph.invoke.side_effect = side_effect
        r = client.post("/api/agent/run", json={
            "screenshot": MOCK_B64,
            "start_lat": 35.6595,
            "start_lon": 139.7005,
            "max_iterations": 5,
        })

    assert r.status_code == 200
    assert mock_graph.invoke.call_count == 2
    assert mock_exec.call_count == 1


def test_run_endpoint_stops_at_max_iterations(client):
    """Always-ROTATE graph should stop after max_iterations."""
    mock_resp = MagicMock()
    mock_resp.content = _TINY_JPEG_BYTES
    mock_resp.raise_for_status = MagicMock()

    with patch("app.geo_graph") as mock_graph, \
         patch("app.execute_action", return_value=(MOCK_B64, GameView(35.0, 139.0, 90))), \
         patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": MOCK_MAPS_KEY}):
        mock_graph.invoke.return_value = MOCK_ROTATE_RESULT
        r = client.post("/api/agent/run", json={
            "screenshot": MOCK_B64,
            "start_lat": 35.6595,
            "start_lon": 139.7005,
            "max_iterations": 3,
        })

    assert r.status_code == 200
    assert mock_graph.invoke.call_count == 3


def test_run_endpoint_response_shape(client):
    with patch("app.geo_graph") as mock_graph:
        mock_graph.invoke.return_value = MOCK_GUESS_RESULT
        r = client.post("/api/agent/run", json={
            "screenshot": MOCK_B64,
            "start_lat": 0.0,
            "start_lon": 0.0,
        })
    data = r.json()
    for key in ["final_guess", "belief_state", "iterations_used", "final_view"]:
        assert key in data, f"Missing key: {key}"


def test_run_endpoint_fresh_specialist_outputs_each_iteration(client):
    """Each graph invocation should receive empty specialist_outputs."""
    received_states = []
    def capture_state(state):
        received_states.append(dict(state))
        return MOCK_GUESS_RESULT

    with patch("app.geo_graph") as mock_graph:
        mock_graph.invoke.side_effect = capture_state
        client.post("/api/agent/run", json={
            "screenshot": MOCK_B64, "start_lat": 0.0, "start_lon": 0.0,
        })

    assert received_states[0]["specialist_outputs"] == {}


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestActionLoopRealAPIs:

    @pytest.fixture(autouse=True)
    def require_keys(self):
        from dotenv import load_dotenv
        load_dotenv()
        if not os.getenv("GOOGLE_MAPS_API_KEY", ""):
            pytest.skip("GOOGLE_MAPS_API_KEY not set")
        if not os.getenv("GEMINI_API_KEY", ""):
            pytest.skip("GEMINI_API_KEY not set")

    def test_fetch_real_streetview(self):
        """Fetch a real Street View frame for Shibuya Crossing."""
        from graphs.action_executor import GameView, fetch_streetview_screenshot
        from dotenv import load_dotenv
        load_dotenv()
        view = GameView(35.6595, 139.7005, heading=0)
        result = fetch_streetview_screenshot(view, os.getenv("GOOGLE_MAPS_API_KEY"))
        assert result.startswith("data:image/jpeg;base64,")
        assert len(result) > 1000  # real image, not empty

    def test_rotate_fetches_different_frame(self):
        """After rotating 180°, the fetched frame should differ from the original."""
        from graphs.action_executor import GameView, fetch_streetview_screenshot
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("GOOGLE_MAPS_API_KEY")
        view = GameView(35.6595, 139.7005, heading=0)
        frame_0 = fetch_streetview_screenshot(view, key)
        frame_180 = fetch_streetview_screenshot(view.rotate(180), key)
        assert frame_0 != frame_180
