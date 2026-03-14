"""
Tests for the Gemini vision specialist nodes.

Unit tests  — mock call_gemini_vision, no API key needed, run always.
Integration tests — hit real Gemini API, require GEMINI_API_KEY env var,
                    run with: pytest -m integration
"""
import base64
import json
import os
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Minimal 1x1 white JPEG (same as stage 1 tests)
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


def _base_state():
    from graphs.state import GeoState
    return GeoState(
        screenshot=MOCK_B64,
        iteration=0,
        max_iterations=5,
        specialist_outputs={},
        belief_state=[],
        action={},
        final_guess=None,
        error=None,
    )


# ---------------------------------------------------------------------------
# Mock responses for each specialist
# ---------------------------------------------------------------------------

MOCK_RESPONSES = {
    "text_language": json.dumps({
        "agent": "text_language",
        "detected_scripts": ["Cyrillic"],
        "language_hints": ["Russian", "Ukrainian"],
        "place_names": ["Москва"],
        "confidence": 0.85,
        "evidence": "Cyrillic script on shop signs with .ru domain suffix visible",
    }),
    "architecture": json.dumps({
        "agent": "architecture",
        "building_styles": ["Soviet-era residential"],
        "materials": ["concrete", "brick"],
        "urban_density": "high",
        "street_furniture": ["utility poles", "bus shelter"],
        "national_flags_or_symbols": [],
        "confidence": 0.7,
        "evidence": "Khrushchevka-style apartment blocks with concrete panel construction",
    }),
    "climate_terrain": json.dumps({
        "agent": "climate_terrain",
        "climate_zone": "continental",
        "terrain_type": "flat_plain",
        "sky_and_light": "overcast grey sky, diffuse light suggesting northern latitude",
        "soil_color": "dark brown",
        "road_surface": "asphalt",
        "confidence": 0.6,
        "evidence": "Grey overcast sky and bare deciduous trees consistent with Eastern Europe in winter",
    }),
    "vegetation": json.dumps({
        "agent": "vegetation",
        "vegetation_type": "temperate deciduous",
        "biome": "temperate_forest",
        "notable_species": ["birch trees", "oak"],
        "season_hints": "bare branches indicate late autumn or winter",
        "confidence": 0.65,
        "evidence": "Birch trees with white bark and bare branches typical of Eastern European winter",
    }),
    "road_infra": json.dumps({
        "agent": "road_infra",
        "driving_side": "right",
        "road_markings": "white dashed centre line, Soviet-style road layout",
        "sign_shapes_colors": "white rectangular signs with black text",
        "vehicle_types": ["Lada", "VAZ"],
        "road_quality": "moderate",
        "camera_rig_clues": [],
        "confidence": 0.75,
        "evidence": "Right-hand traffic with Soviet-era road sign conventions and domestic vehicles",
    }),
}


# ---------------------------------------------------------------------------
# Unit tests — gemini_vision helpers
# ---------------------------------------------------------------------------

class TestGeminiVisionHelpers:
    def test_extract_b64_from_data_url(self):
        from graphs.nodes.gemini_vision import _extract_b64
        b64, mime = _extract_b64("data:image/jpeg;base64,AAAA")
        assert b64 == "AAAA"
        assert mime == "image/jpeg"

    def test_extract_b64_from_raw_base64(self):
        from graphs.nodes.gemini_vision import _extract_b64
        b64, mime = _extract_b64("AAABBBCCC")
        assert b64 == "AAABBBCCC"
        assert mime == "image/jpeg"

    def test_parse_json_plain(self):
        from graphs.nodes.gemini_vision import parse_json_response
        result = parse_json_response('{"key": "value"}')
        assert result["key"] == "value"

    def test_parse_json_with_markdown_fence(self):
        from graphs.nodes.gemini_vision import parse_json_response
        result = parse_json_response('```json\n{"key": "value"}\n```')
        assert result["key"] == "value"

    def test_parse_json_with_plain_fence(self):
        from graphs.nodes.gemini_vision import parse_json_response
        result = parse_json_response('```\n{"key": "value"}\n```')
        assert result["key"] == "value"

    def test_call_gemini_raises_without_api_key(self):
        from graphs.nodes.gemini_vision import call_gemini_vision
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            call_gemini_vision("prompt", MOCK_B64, api_key="")


# ---------------------------------------------------------------------------
# Unit tests — specialist nodes (mocked Gemini)
# ---------------------------------------------------------------------------

SPECIALIST_CASES = [
    ("text_language_node",  "text_language",  ["agent", "detected_scripts", "language_hints", "place_names", "confidence", "evidence"]),
    ("architecture_node",   "architecture",   ["agent", "building_styles", "materials", "urban_density", "confidence", "evidence"]),
    ("climate_terrain_node","climate_terrain", ["agent", "climate_zone", "terrain_type", "confidence", "evidence"]),
    ("vegetation_node",     "vegetation",     ["agent", "vegetation_type", "biome", "confidence", "evidence"]),
    ("road_infra_node",     "road_infra",     ["agent", "driving_side", "road_quality", "confidence", "evidence"]),
]


@pytest.mark.parametrize("func_name,agent_key,required_fields", SPECIALIST_CASES)
def test_specialist_node_mocked(func_name, agent_key, required_fields):
    """Each node returns correct schema when Gemini is mocked."""
    import graphs.nodes.specialists as s
    node_fn = getattr(s, func_name)
    mock_response = MOCK_RESPONSES[agent_key]

    with patch("graphs.nodes.specialists.call_gemini_vision", return_value=mock_response):
        result = node_fn(_base_state())

    assert "specialist_outputs" in result
    output = result["specialist_outputs"][agent_key]
    for field in required_fields:
        assert field in output, f"Missing '{field}' in {agent_key} output"
    assert 0.0 <= output["confidence"] <= 1.0


@pytest.mark.parametrize("func_name,agent_key,_", SPECIALIST_CASES)
def test_specialist_node_graceful_on_bad_json(func_name, agent_key, _):
    """Node returns fallback with error key when Gemini returns unparseable output."""
    import graphs.nodes.specialists as s
    node_fn = getattr(s, func_name)

    with patch("graphs.nodes.specialists.call_gemini_vision", return_value="not valid json {{"):
        result = node_fn(_base_state())

    output = result["specialist_outputs"][agent_key]
    assert "error" in output
    assert output["confidence"] == 0.0


@pytest.mark.parametrize("func_name,agent_key,_", SPECIALIST_CASES)
def test_specialist_node_graceful_on_api_error(func_name, agent_key, _):
    """Node returns fallback when Gemini API raises an exception."""
    import graphs.nodes.specialists as s
    node_fn = getattr(s, func_name)

    with patch("graphs.nodes.specialists.call_gemini_vision", side_effect=Exception("timeout")):
        result = node_fn(_base_state())

    output = result["specialist_outputs"][agent_key]
    assert "error" in output
    assert output["confidence"] == 0.0


def test_full_graph_with_mocked_specialists():
    """Full graph produces all 5 specialist outputs when Gemini is mocked."""
    import json
    from graphs.geoguessr_graph import geo_graph

    fusion_mock = json.dumps({
        "belief_state": [{"country": "Russia", "lat": 55.75, "lon": 37.62, "confidence": 0.8, "evidence": "mocked"}],
        "decision": "GUESS", "action": {"type": "GUESS", "lat": 55.75, "lon": 37.62},
        "reasoning": "mocked", "top_confidence": 0.8,
    })

    def mock_specialist(prompt, screenshot, api_key, model=None):
        for key, resp in MOCK_RESPONSES.items():
            if f'"agent": "{key}"' in resp and key in prompt.lower().replace(" ", "_"):
                return resp
        return MOCK_RESPONSES["text_language"]

    with patch("graphs.nodes.specialists.call_gemini_vision", side_effect=mock_specialist), \
         patch("graphs.nodes.fusion.call_gemini_vision", return_value=fusion_mock):
        result = geo_graph.invoke(_base_state())

    for agent in ["text_language", "architecture", "climate_terrain", "vegetation", "road_infra"]:
        assert agent in result["specialist_outputs"], f"Missing specialist: {agent}"


# ---------------------------------------------------------------------------
# Integration tests — hit real Gemini API
# Run with: pytest -m integration
# Requires: GEMINI_API_KEY set in environment or .env
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSpecialistsRealGemini:
    """
    These tests call the real Gemini API. They verify that:
    - The model returns parseable JSON
    - Required fields are present
    - Confidence is a valid float
    They do NOT assert specific countries/languages since model outputs vary.
    """

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            pytest.skip("GEMINI_API_KEY not set — skipping integration test")

    def test_text_language_real(self):
        from graphs.nodes.specialists import text_language_node
        result = text_language_node(_base_state())
        out = result["specialist_outputs"]["text_language"]
        assert "detected_scripts" in out
        assert isinstance(out["confidence"], (int, float))
        assert 0.0 <= out["confidence"] <= 1.0

    def test_architecture_real(self):
        from graphs.nodes.specialists import architecture_node
        result = architecture_node(_base_state())
        out = result["specialist_outputs"]["architecture"]
        assert "building_styles" in out
        assert isinstance(out["confidence"], (int, float))

    def test_climate_terrain_real(self):
        from graphs.nodes.specialists import climate_terrain_node
        result = climate_terrain_node(_base_state())
        out = result["specialist_outputs"]["climate_terrain"]
        assert "climate_zone" in out
        assert isinstance(out["confidence"], (int, float))

    def test_vegetation_real(self):
        from graphs.nodes.specialists import vegetation_node
        result = vegetation_node(_base_state())
        out = result["specialist_outputs"]["vegetation"]
        assert "biome" in out
        assert isinstance(out["confidence"], (int, float))

    def test_road_infra_real(self):
        from graphs.nodes.specialists import road_infra_node
        result = road_infra_node(_base_state())
        out = result["specialist_outputs"]["road_infra"]
        assert "driving_side" in out
        assert isinstance(out["confidence"], (int, float))
