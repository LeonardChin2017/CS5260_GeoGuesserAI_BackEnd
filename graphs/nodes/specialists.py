"""
Real specialist nodes — each calls Gemini vision with a focused prompt
and returns structured JSON evidence for the fusion/planner node.

Each node only writes its own key into specialist_outputs.
The Annotated reducer in GeoState merges all parallel outputs automatically.
"""
import os

from graphs.nodes.gemini_vision import call_gemini_vision, parse_json_response
from graphs.state import GeoState

# ---------------------------------------------------------------------------
# Prompts — kept focused so the model returns clean, parseable JSON
# ---------------------------------------------------------------------------

_TEXT_LANGUAGE_PROMPT = """You are a specialist in identifying text, language, and writing systems from street-level images.

Analyze this image and identify any visible text, scripts, or language clues.

Return ONLY a valid JSON object — no markdown, no explanation — with this exact schema:
{
  "agent": "text_language",
  "detected_scripts": ["list of writing systems visible, e.g. Latin, Cyrillic, Arabic, Chinese, Devanagari, Thai, Hebrew, Georgian"],
  "language_hints": ["list of possible languages based on scripts and text patterns"],
  "place_names": ["any readable place names, street names, phone prefixes, or domain suffixes"],
  "confidence": <float 0.0-1.0, how certain you are about the language/region>,
  "evidence": "<one sentence summarising the strongest clue you found>"
}

If no text is visible, return low confidence and empty lists."""

_ARCHITECTURE_PROMPT = """You are a specialist in identifying architectural styles and built environment clues from street-level images.

Analyze this image for building styles, construction materials, and urban features.

Return ONLY a valid JSON object — no markdown, no explanation — with this exact schema:
{
  "agent": "architecture",
  "building_styles": ["list of styles/periods, e.g. Soviet-era, Colonial, Ottoman, Scandinavian modern, South Asian vernacular"],
  "materials": ["dominant construction materials, e.g. brick, concrete, wood, adobe"],
  "urban_density": "<low | medium | high | rural | uninhabited>",
  "street_furniture": ["notable items, e.g. utility poles, letterboxes, bus shelters, bollards"],
  "national_flags_or_symbols": ["any visible national or regional symbols"],
  "confidence": <float 0.0-1.0>,
  "evidence": "<one sentence summarising the strongest architectural clue>"
}"""

_CLIMATE_TERRAIN_PROMPT = """You are a specialist in identifying climate zones and terrain features from street-level images.

Analyze this image for environmental and geographical clues.

Return ONLY a valid JSON object — no markdown, no explanation — with this exact schema:
{
  "agent": "climate_terrain",
  "climate_zone": "<one of: tropical, arid, semi-arid, mediterranean, temperate, continental, subarctic, polar, unknown>",
  "terrain_type": "<one of: coastal, mountainous, flat_plain, hilly, desert, urban, unknown>",
  "sky_and_light": "<description of sky colour, sun angle, shadows that suggest latitude or season>",
  "soil_color": "<visible soil or ground color if any>",
  "road_surface": "<asphalt, dirt, cobblestone, gravel, etc.>",
  "confidence": <float 0.0-1.0>,
  "evidence": "<one sentence summarising the strongest climate/terrain clue>"
}"""

_VEGETATION_PROMPT = """You are a specialist in identifying vegetation types and biomes from street-level images.

Analyze this image for plants, trees, and ground cover.

Return ONLY a valid JSON object — no markdown, no explanation — with this exact schema:
{
  "agent": "vegetation",
  "vegetation_type": "<dominant type, e.g. tropical broadleaf, boreal conifer, savanna grass, mediterranean shrub, temperate deciduous, agricultural crops, none>",
  "biome": "<one of: tropical_rainforest, savanna, desert, mediterranean, temperate_forest, boreal_forest, tundra, grassland, urban, wetland, unknown>",
  "notable_species": ["any identifiable plant species or types, e.g. palm trees, eucalyptus, rice paddies, baobab"],
  "season_hints": "<seasonal clues from vegetation, e.g. bare deciduous = winter/autumn, lush green = summer>",
  "confidence": <float 0.0-1.0>,
  "evidence": "<one sentence summarising the strongest vegetation clue>"
}"""

_ROAD_INFRA_PROMPT = """You are a specialist in identifying road infrastructure, traffic conventions, and vehicle clues from street-level images.

Analyze this image for road markings, signs, and vehicles.

Return ONLY a valid JSON object — no markdown, no explanation — with this exact schema:
{
  "agent": "road_infra",
  "driving_side": "<left | right | unknown>",
  "road_markings": "<description of lane markings, e.g. yellow centre line = North America, white dashed = Europe>",
  "sign_shapes_colors": "<description of road sign shapes and colors visible>",
  "vehicle_types": ["visible vehicle types or makes that suggest a region"],
  "road_quality": "<poor | moderate | good | excellent>",
  "camera_rig_clues": ["any Google Street View camera rig clues, e.g. car roof colour, bicycle mount>"],
  "confidence": <float 0.0-1.0>,
  "evidence": "<one sentence summarising the strongest infrastructure clue>"
}"""


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "")


def _run_specialist(agent_key: str, prompt: str, state: GeoState, fallback: dict) -> dict:
    """
    Call Gemini vision, parse JSON, and return specialist_outputs update.
    On any error, returns the fallback dict with error info.
    """
    try:
        raw = call_gemini_vision(prompt, state["screenshot"], _get_api_key())
        output = parse_json_response(raw)
        # Ensure required fields are present
        output.setdefault("agent", agent_key)
        output.setdefault("confidence", 0.0)
        output.setdefault("evidence", "")
    except Exception as exc:
        output = {**fallback, "error": str(exc), "confidence": 0.0}

    return {"specialist_outputs": {agent_key: output}}


# ---------------------------------------------------------------------------
# Specialist nodes
# ---------------------------------------------------------------------------

def text_language_node(state: GeoState) -> dict:
    return _run_specialist(
        "text_language",
        _TEXT_LANGUAGE_PROMPT,
        state,
        fallback={
            "agent": "text_language",
            "detected_scripts": [],
            "language_hints": [],
            "place_names": [],
            "evidence": "Analysis failed",
        },
    )


def architecture_node(state: GeoState) -> dict:
    return _run_specialist(
        "architecture",
        _ARCHITECTURE_PROMPT,
        state,
        fallback={
            "agent": "architecture",
            "building_styles": [],
            "materials": [],
            "urban_density": "unknown",
            "street_furniture": [],
            "national_flags_or_symbols": [],
            "evidence": "Analysis failed",
        },
    )


def climate_terrain_node(state: GeoState) -> dict:
    return _run_specialist(
        "climate_terrain",
        _CLIMATE_TERRAIN_PROMPT,
        state,
        fallback={
            "agent": "climate_terrain",
            "climate_zone": "unknown",
            "terrain_type": "unknown",
            "sky_and_light": "",
            "soil_color": "",
            "road_surface": "unknown",
            "evidence": "Analysis failed",
        },
    )


def vegetation_node(state: GeoState) -> dict:
    return _run_specialist(
        "vegetation",
        _VEGETATION_PROMPT,
        state,
        fallback={
            "agent": "vegetation",
            "vegetation_type": "unknown",
            "biome": "unknown",
            "notable_species": [],
            "season_hints": "",
            "evidence": "Analysis failed",
        },
    )


def road_infra_node(state: GeoState) -> dict:
    return _run_specialist(
        "road_infra",
        _ROAD_INFRA_PROMPT,
        state,
        fallback={
            "agent": "road_infra",
            "driving_side": "unknown",
            "road_markings": "",
            "sign_shapes_colors": "",
            "vehicle_types": [],
            "road_quality": "unknown",
            "camera_rig_clues": [],
            "evidence": "Analysis failed",
        },
    )
