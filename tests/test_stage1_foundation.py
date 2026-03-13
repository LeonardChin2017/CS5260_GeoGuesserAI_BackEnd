"""
Stage 1 tests — Foundation & State Schema.
Verifies that:
  1. GeoState schema is correct
  2. Each stub node returns the right keys
  3. The full graph runs end-to-end and returns expected structure
  4. The /api/agent/analyze endpoint works correctly
"""
import base64
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Minimal 1x1 white JPEG as a test screenshot (no external file needed)
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
# 1. State schema
# ---------------------------------------------------------------------------

def test_state_schema_all_keys():
    from graphs.state import GeoState
    state: GeoState = {
        "screenshot": MOCK_B64,
        "iteration": 0,
        "max_iterations": 5,
        "specialist_outputs": {},
        "belief_state": [],
        "action": {},
        "action_history": [],
        "final_guess": None,
        "error": None,
    }
    assert state["iteration"] == 0
    assert state["max_iterations"] == 5
    assert state["specialist_outputs"] == {}
    assert state["final_guess"] is None


# ---------------------------------------------------------------------------
# 2. Stub nodes — correct output keys
# ---------------------------------------------------------------------------

def _base_state():
    from graphs.state import GeoState
    return GeoState(
        screenshot=MOCK_B64,
        iteration=0,
        max_iterations=5,
        specialist_outputs={},
        belief_state=[],
        action={},
        action_history=[],
        final_guess=None,
        error=None,
    )


SPECIALIST_STUBS = [
    ("text_language",   "text_language",   ["agent", "detected_scripts", "language_hints", "confidence", "evidence"]),
    ("architecture",    "architecture",    ["agent", "building_styles", "confidence", "evidence"]),
    ("climate_terrain", "climate_terrain", ["agent", "climate_zone", "confidence", "evidence"]),
    ("vegetation",      "vegetation",      ["agent", "vegetation_type", "confidence", "evidence"]),
    ("road_infra",      "road_infra",      ["agent", "driving_side", "confidence", "evidence"]),
]


@pytest.mark.parametrize("func_name,key,required_fields", SPECIALIST_STUBS)
def test_stub_node_output(func_name, key, required_fields):
    import graphs.nodes.stubs as stubs
    fn = getattr(stubs, f"{func_name}_stub")
    result = fn(_base_state())
    assert "specialist_outputs" in result
    output = result["specialist_outputs"][key]
    for field in required_fields:
        assert field in output, f"Missing field '{field}' in {key} output"
    assert 0.0 <= output["confidence"] <= 1.0


def test_ingest_node_passes_with_screenshot():
    from graphs.nodes.stubs import ingest_node
    result = ingest_node(_base_state())
    assert result.get("error") is None


def test_ingest_node_fails_without_screenshot():
    from graphs.nodes.stubs import ingest_node
    state = _base_state()
    state["screenshot"] = ""
    result = ingest_node(state)
    assert result["error"] is not None


# ---------------------------------------------------------------------------
# 3. Full graph — end-to-end
# ---------------------------------------------------------------------------

def test_graph_runs_end_to_end():
    from graphs.geoguessr_graph import geo_graph
    result = geo_graph.invoke(_base_state())
    assert "action" in result
    assert "belief_state" in result
    assert "action_history" in result
    assert "specialist_outputs" in result


def test_graph_action_is_valid_type():
    from graphs.geoguessr_graph import geo_graph
    result = geo_graph.invoke(_base_state())
    assert result["action"]["type"] in ("GUESS", "ROTATE", "MOVE")


def test_graph_all_specialists_present():
    from graphs.geoguessr_graph import geo_graph
    result = geo_graph.invoke(_base_state())
    for agent in ["text_language", "architecture", "climate_terrain", "vegetation", "road_infra"]:
        assert agent in result["specialist_outputs"], f"Missing specialist: {agent}"


def test_graph_commits_at_budget():
    """At max_iterations=1, fusion should commit a GUESS immediately."""
    from graphs.geoguessr_graph import geo_graph
    state = _base_state()
    state["max_iterations"] = 1
    result = geo_graph.invoke(state)
    assert result["action"]["type"] == "GUESS"
    assert result["final_guess"] is not None


def test_graph_explores_before_budget():
    """With budget > 1 and iteration=0, fusion should explore (ROTATE/MOVE)."""
    from graphs.geoguessr_graph import geo_graph
    state = _base_state()
    state["max_iterations"] = 5
    state["iteration"] = 0
    result = geo_graph.invoke(state)
    assert result["action"]["type"] in ("ROTATE", "MOVE")
    assert result["final_guess"] is None


def test_graph_action_history_grows():
    from graphs.geoguessr_graph import geo_graph
    result = geo_graph.invoke(_base_state())
    assert len(result["action_history"]) >= 1


# ---------------------------------------------------------------------------
# 4. API endpoint — /api/agent/analyze
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from app import app
    return TestClient(app)


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_analyze_endpoint_returns_200(client):
    r = client.post("/api/agent/analyze", json={"screenshot": MOCK_B64, "max_iterations": 5})
    assert r.status_code == 200


def test_analyze_endpoint_response_shape(client):
    r = client.post("/api/agent/analyze", json={"screenshot": MOCK_B64, "max_iterations": 5})
    data = r.json()
    for key in ["belief_state", "action", "action_history", "specialist_outputs", "iteration"]:
        assert key in data, f"Missing key '{key}' in response"


def test_analyze_endpoint_action_type(client):
    r = client.post("/api/agent/analyze", json={"screenshot": MOCK_B64, "max_iterations": 5})
    assert r.json()["action"]["type"] in ("GUESS", "ROTATE", "MOVE")


def test_analyze_endpoint_commits_at_max_1(client):
    r = client.post("/api/agent/analyze", json={"screenshot": MOCK_B64, "max_iterations": 1})
    data = r.json()
    assert data["action"]["type"] == "GUESS"
    assert data["final_guess"] is not None
