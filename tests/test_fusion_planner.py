"""
Tests for the fusion/planner node.

Unit tests  — mock call_gemini_vision, no API key needed.
Integration — hit real Gemini API, require GEMINI_API_KEY, run with: pytest -m integration
"""
import base64
import json
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


# Realistic specialist outputs for a Russia scene
HIGH_CONFIDENCE_SPECIALISTS = {
    "text_language": {
        "agent": "text_language", "detected_scripts": ["Cyrillic"],
        "language_hints": ["Russian"], "place_names": ["Москва"],
        "confidence": 0.9, "evidence": "Cyrillic script on signs",
    },
    "architecture": {
        "agent": "architecture", "building_styles": ["Soviet-era"],
        "materials": ["concrete"], "urban_density": "high",
        "street_furniture": ["utility poles"], "national_flags_or_symbols": [],
        "confidence": 0.75, "evidence": "Khrushchevka apartment blocks",
    },
    "climate_terrain": {
        "agent": "climate_terrain", "climate_zone": "continental",
        "terrain_type": "flat_plain", "sky_and_light": "overcast grey",
        "soil_color": "dark brown", "road_surface": "asphalt",
        "confidence": 0.6, "evidence": "Grey sky, bare trees, Eastern Europe winter",
    },
    "vegetation": {
        "agent": "vegetation", "vegetation_type": "temperate deciduous",
        "biome": "temperate_forest", "notable_species": ["birch"],
        "season_hints": "bare branches = winter", "confidence": 0.65,
        "evidence": "Birch trees typical of Eastern Europe",
    },
    "road_infra": {
        "agent": "road_infra", "driving_side": "right",
        "road_markings": "white dashed", "sign_shapes_colors": "white rectangular",
        "vehicle_types": ["Lada"], "road_quality": "moderate",
        "camera_rig_clues": [], "confidence": 0.75,
        "evidence": "Right-hand traffic, Soviet road signs",
    },
}

# Vague scene — low confidence across the board
LOW_CONFIDENCE_SPECIALISTS = {
    "text_language": {
        "agent": "text_language", "detected_scripts": [],
        "language_hints": [], "place_names": [],
        "confidence": 0.05, "evidence": "No text visible",
    },
    "architecture": {
        "agent": "architecture", "building_styles": ["generic"],
        "materials": ["concrete"], "urban_density": "low",
        "street_furniture": [], "national_flags_or_symbols": [],
        "confidence": 0.1, "evidence": "Generic low-rise buildings",
    },
    "climate_terrain": {
        "agent": "climate_terrain", "climate_zone": "temperate",
        "terrain_type": "flat_plain", "sky_and_light": "cloudy",
        "soil_color": "unknown", "road_surface": "asphalt",
        "confidence": 0.15, "evidence": "Temperate climate, could be many regions",
    },
    "vegetation": {
        "agent": "vegetation", "vegetation_type": "temperate deciduous",
        "biome": "temperate_forest", "notable_species": [],
        "season_hints": "green leaves = summer", "confidence": 0.1,
        "evidence": "Generic temperate vegetation",
    },
    "road_infra": {
        "agent": "road_infra", "driving_side": "unknown",
        "road_markings": "unclear", "sign_shapes_colors": "unclear",
        "vehicle_types": [], "road_quality": "moderate",
        "camera_rig_clues": [], "confidence": 0.1,
        "evidence": "Cannot determine driving side",
    },
}

# Mock Gemini responses
MOCK_FUSION_GUESS = json.dumps({
    "belief_state": [
        {"country": "Russia", "region": "Moscow Oblast", "lat": 55.75, "lon": 37.62,
         "confidence": 0.85, "evidence": "Cyrillic + Soviet architecture + birch trees all converge on Russia"},
        {"country": "Ukraine", "region": None, "lat": 50.45, "lon": 30.52,
         "confidence": 0.15, "evidence": "Similar Cyrillic script but lower confidence"},
    ],
    "decision": "GUESS",
    "action": {"type": "GUESS", "lat": 55.75, "lon": 37.62},
    "reasoning": "Cyrillic script is decisive. Soviet architecture and birch trees corroborate Russia. Confidence 0.85 exceeds threshold.",
    "top_confidence": 0.85,
})

MOCK_FUSION_ROTATE = json.dumps({
    "belief_state": [
        {"country": "Eastern Europe", "region": None, "lat": 50.0, "lon": 30.0,
         "confidence": 0.35, "evidence": "Temperate vegetation and right-hand traffic narrow to Europe"},
    ],
    "decision": "ROTATE",
    "action": {"type": "ROTATE", "degrees": 90},
    "reasoning": "No text visible and low confidence. Rotating 90 degrees may reveal road signs or storefronts.",
    "top_confidence": 0.35,
})


def _make_state(iteration=0, max_iterations=5, specialists=None, belief_state=None, screenshot=None):
    from graphs.state import GeoState
    return GeoState(
        screenshot=screenshot or MOCK_B64,
        iteration=iteration,
        max_iterations=max_iterations,
        specialist_outputs=specialists or HIGH_CONFIDENCE_SPECIALISTS,
        belief_state=belief_state or [],
        action={},
        final_guess={},
        error='',
    )


# ---------------------------------------------------------------------------
# Unit tests — fusion node logic (mocked Gemini)
# ---------------------------------------------------------------------------

def test_fusion_commits_on_high_confidence():
    from graphs.nodes.fusion import fusion_planner_node
    with patch("graphs.nodes.fusion.call_gemini_vision", return_value=MOCK_FUSION_GUESS):
        result = fusion_planner_node(_make_state())
    assert result["action"]["type"] == "GUESS"
    assert len(result["final_guess"]) > 0
    assert result["final_guess"]["lat"] == 55.75
    assert result["final_guess"]["lon"] == 37.62


def test_fusion_explores_on_low_confidence():
    from graphs.nodes.fusion import fusion_planner_node
    with patch("graphs.nodes.fusion.call_gemini_vision", return_value=MOCK_FUSION_ROTATE):
        result = fusion_planner_node(_make_state(specialists=LOW_CONFIDENCE_SPECIALISTS))
    assert result["action"]["type"] in ("ROTATE", "MOVE")
    assert len(result["final_guess"]) <= 0


def test_fusion_forces_guess_at_budget():
    """At max_iterations-1, must commit a GUESS without calling Gemini."""
    from graphs.nodes.fusion import fusion_planner_node
    state = _make_state(
        iteration=4,
        max_iterations=5,
        belief_state=[{"country": "Russia", "lat": 55.75, "lon": 37.62, "confidence": 0.5}],
    )
    # Gemini should NOT be called — patch it to raise to confirm
    with patch("graphs.nodes.fusion.call_gemini_vision", side_effect=Exception("should not be called")):
        result = fusion_planner_node(state)
    assert result["action"]["type"] == "GUESS"
    assert result["final_guess"]["forced"] is True


def test_fusion_belief_state_sorted_by_confidence():
    from graphs.nodes.fusion import fusion_planner_node
    with patch("graphs.nodes.fusion.call_gemini_vision", return_value=MOCK_FUSION_GUESS):
        result = fusion_planner_node(_make_state())
    beliefs = result["belief_state"]
    assert len(beliefs) >= 2
    for i in range(len(beliefs) - 1):
        assert beliefs[i]["confidence"] >= beliefs[i + 1]["confidence"]


def test_fusion_belief_state_has_required_fields():
    from graphs.nodes.fusion import fusion_planner_node
    with patch("graphs.nodes.fusion.call_gemini_vision", return_value=MOCK_FUSION_GUESS):
        result = fusion_planner_node(_make_state())
    for entry in result["belief_state"]:
        for field in ["country", "lat", "lon", "confidence", "evidence"]:
            assert field in entry, f"Missing '{field}' in belief_state entry"


def test_fusion_iteration_increments():
    from graphs.nodes.fusion import fusion_planner_node
    with patch("graphs.nodes.fusion.call_gemini_vision", return_value=MOCK_FUSION_GUESS):
        result = fusion_planner_node(_make_state(iteration=2))
    assert result["iteration"] == 3


def test_fusion_iteration_does_not_exceed_max_iterations():
    from graphs.nodes.fusion import fusion_planner_node
    with patch("graphs.nodes.fusion.call_gemini_vision", return_value=MOCK_FUSION_GUESS):
        result = fusion_planner_node(_make_state(iteration=0, max_iterations=0))
    assert result["iteration"] == 0


def test_fusion_graceful_on_api_error():
    from graphs.nodes.fusion import fusion_planner_node
    with patch("graphs.nodes.fusion.call_gemini_vision", side_effect=Exception("timeout")):
        result = fusion_planner_node(_make_state())
    assert result["action"]["type"] == "GUESS"
    assert "error" in result
    assert len(result["error"]) > 0


def test_fusion_graceful_on_bad_json():
    from graphs.nodes.fusion import fusion_planner_node
    with patch("graphs.nodes.fusion.call_gemini_vision", return_value="not valid json {{"):
        result = fusion_planner_node(_make_state())
    assert result["action"]["type"] == "GUESS"
    assert len(result["error"]) > 0


def test_fusion_normalises_unknown_action_type():
    """If model returns an unexpected action type, default to GUESS."""
    from graphs.nodes.fusion import fusion_planner_node
    bad_response = json.dumps({
        "belief_state": [{"country": "Russia", "lat": 55.75, "lon": 37.62, "confidence": 0.8, "evidence": "test"}],
        "decision": "TELEPORT",
        "action": {"type": "TELEPORT"},
        "reasoning": "test", "top_confidence": 0.8,
    })
    with patch("graphs.nodes.fusion.call_gemini_vision", return_value=bad_response):
        result = fusion_planner_node(_make_state())
    assert result["action"]["type"] == "GUESS"


# ---------------------------------------------------------------------------
# Full graph test with mocked specialists + fusion
# ---------------------------------------------------------------------------

def test_full_graph_with_real_fusion_mocked():
    """End-to-end: specialists mocked, fusion mocked, verify complete state shape."""
    from graphs.geoguessr_graph import geo_graph
    from tests.test_gemini_specialists import MOCK_RESPONSES

    call_count = {"n": 0}

    def mock_gemini(prompt, screenshot, api_key, model=None):
        call_count["n"] += 1
        # Last call is fusion (longest prompt)
        for key, resp in MOCK_RESPONSES.items():
            if f'"agent": "{key}"' in resp and key.replace("_", " ") in prompt.lower():
                return resp
        return MOCK_FUSION_GUESS  # fusion call

    with patch("graphs.nodes.specialists.call_gemini_vision", side_effect=mock_gemini):
        with patch("graphs.nodes.fusion.call_gemini_vision", return_value=MOCK_FUSION_GUESS):
            result = geo_graph.invoke(_make_state())

    assert result["action"]["type"] in ("GUESS", "ROTATE", "MOVE")
    assert result["belief_state"]
    assert result["iteration"] == 1


# ---------------------------------------------------------------------------
# Integration tests — real Gemini API
# Run with: pytest -m integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFusionRealGemini:
    def test_fusion_returns_valid_structure(self):
        from graphs.nodes.fusion import fusion_planner_node
        result = fusion_planner_node(_make_state(specialists=HIGH_CONFIDENCE_SPECIALISTS))
        assert result["action"]["type"] in ("GUESS", "ROTATE", "MOVE")
        assert isinstance(result["belief_state"], list)
        assert result["iteration"] == 1

    def test_fusion_belief_state_has_coordinates(self):
        from graphs.nodes.fusion import fusion_planner_node
        result = fusion_planner_node(_make_state(specialists=HIGH_CONFIDENCE_SPECIALISTS))
        if result["belief_state"]:
            top = result["belief_state"][0]
            assert isinstance(top.get("lat"), (int, float))
            assert isinstance(top.get("lon"), (int, float))