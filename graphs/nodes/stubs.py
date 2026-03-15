"""
Stage 1 stub nodes — return mock structured data for testing.

Each specialist node returns ONLY its own key in specialist_outputs.
The Annotated reducer in GeoState merges all parallel outputs automatically.
"""
from graphs.state import GeoState
from graphs.nodes.ingest import ingest_node  # re-exported for backward compat


def text_language_stub(state: GeoState) -> dict:
    return {
        "specialist_outputs": {
            "text_language": {
                "agent": "text_language",
                "detected_scripts": ["Latin"],
                "language_hints": ["Unknown"],
                "place_names": [],
                "confidence": 0.1,
                "evidence": "[stub] No real analysis yet",
            }
        }
    }


def architecture_stub(state: GeoState) -> dict:
    return {
        "specialist_outputs": {
            "architecture": {
                "agent": "architecture",
                "building_styles": ["Unknown"],
                "urban_density": "unknown",
                "confidence": 0.1,
                "evidence": "[stub] No real analysis yet",
            }
        }
    }


def climate_terrain_stub(state: GeoState) -> dict:
    return {
        "specialist_outputs": {
            "climate_terrain": {
                "agent": "climate_terrain",
                "climate_zone": "unknown",
                "terrain_type": "unknown",
                "confidence": 0.1,
                "evidence": "[stub] No real analysis yet",
            }
        }
    }


def vegetation_stub(state: GeoState) -> dict:
    return {
        "specialist_outputs": {
            "vegetation": {
                "agent": "vegetation",
                "vegetation_type": "unknown",
                "biome": "unknown",
                "confidence": 0.1,
                "evidence": "[stub] No real analysis yet",
            }
        }
    }


def road_infra_stub(state: GeoState) -> dict:
    return {
        "specialist_outputs": {
            "road_infra": {
                "agent": "road_infra",
                "driving_side": "unknown",
                "road_quality": "unknown",
                "confidence": 0.1,
                "evidence": "[stub] No real analysis yet",
            }
        }
    }


def fusion_planner_stub(state: GeoState) -> dict:
    """
    Stub fusion node: always returns GUESS with a default location.
    Real version will weigh evidence and decide explore vs commit.
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 5)

    if iteration >= max_iter - 1:
        action = {"type": "GUESS", "lat": 0.0, "lon": 0.0}
        final_guess = {"lat": 0.0, "lon": 0.0, "confidence": 0.1}
    else:
        action = {"type": "ROTATE", "degrees": 90}
        final_guess = {}

    belief_state = [
        {
            "country": "Unknown",
            "region": None,
            "lat": 0.0,
            "lon": 0.0,
            "confidence": 0.1,
            "evidence": "[stub] Fusion not implemented yet",
        }
    ]

    return {
        "belief_state": belief_state,
        "action": action,
        "final_guess": final_guess,
        "iteration": iteration + 1,
    }
