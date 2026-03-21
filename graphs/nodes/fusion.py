"""
Fusion/Planner node — the decision-making core of the pipeline.

Receives all specialist outputs, synthesises evidence with weighted reasoning,
produces a ranked belief state of candidate locations, and decides:
  - GUESS(lat, lon)  if confidence is high enough or iteration budget is exhausted
  - ROTATE(degrees)  if turning the view may reveal more diagnostic clues
  - MOVE(forward)    if moving closer to a sign or landmark would help

Evidence weighting hierarchy (encoded in the prompt):
  1. Text & Language  — most diagnostic; a single readable script narrows to <20 countries
  2. Road Infrastructure — driving side alone halves the search space
  3. Architecture     — strong for distinguishing regions within a continent
  4. Climate & Terrain — eliminates hemispheres and broad climate zones
  5. Vegetation       — broad biome narrowing, weakest alone but useful corroboration
"""
import json

from graphs.nodes.gemini_vision import call_gemini_vision, parse_json_response
from graphs.state import GeoState
from util import log_event, GEMINI_API_KEY

# Confidence threshold to commit a guess without exhausting the iteration budget
CONFIDENCE_THRESHOLD = 0.75

_FUSION_PROMPT = """You are the central reasoning agent for a GeoGuessr location solver.

You have received structured evidence from 5 specialist vision agents that analysed the same street-level image.
Your job is to:
1. Synthesise all evidence, resolving any conflicts.
2. Produce a ranked list of candidate locations.
3. Decide whether to commit a guess or request more exploration.

--- EVIDENCE FROM SPECIALISTS ---
{evidence_json}
--- END EVIDENCE ---

Evidence weighting rules (apply in this order):
- Text & Language: HIGHEST weight. A readable script or place name narrows to very few countries.
- Road Infrastructure: HIGH weight. Driving side alone eliminates ~half the world.
- Architecture: MEDIUM weight. Strong within a continent.
- Climate & Terrain: MEDIUM weight. Good for eliminating hemispheres and biomes.
- Vegetation: MEDIUM weight. Useful corroboration but rarely conclusive alone.

When specialists conflict, prefer the higher-confidence output and note the conflict in reasoning.

Return ONLY a valid JSON object — no markdown, no explanation — with this exact schema:
{{
  "belief_state": [
    {{
      "country": "<country name>",
      "region": "<state/province/city if known, else null>",
      "lat": <float, best estimate latitude for this candidate>,
      "lon": <float, best estimate longitude for this candidate>,
      "confidence": <float 0.0-1.0>,
      "evidence": "<one sentence: why this location, citing which specialists agree>"
    }}
  ],
  "decision": "<GUESS | ROTATE | MOVE>",
  "action": {{
    "type": "<GUESS | ROTATE | MOVE>",
    "lat": <float, only present if type=GUESS — use belief_state[0] coordinates>,
    "lon": <float, only present if type=GUESS>,
    "degrees": <int, only present if type=ROTATE — suggest 45, 90, or 180>,
    "direction": "<forward, only present if type=MOVE>"
  }},
  "reasoning": "<2-3 sentences explaining the decision and which evidence was most decisive>",
  "top_confidence": <float, confidence of the best candidate>
}}

Sort belief_state by confidence descending (highest first).
Use GUESS if top_confidence >= {threshold} or if the evidence strongly converges.
Use ROTATE if turning may reveal text, signs, or landmarks not yet visible.
Use MOVE if approaching a visible sign or building would provide decisive clues.
"""


def fusion_planner_node(state: GeoState) -> dict:
    """
    Synthesise specialist evidence and decide: GUESS or EXPLORE.
    Forces GUESS if iteration budget is exhausted.
    Falls back gracefully on API or parse errors.
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 5)
    specialist_outputs = state.get("specialist_outputs", {})

    # --- Force guess at budget ---
    if iteration >= max_iter - 1:
        return _forced_guess(state, specialist_outputs, iteration)

    # --- Call Gemini fusion ---
    try:
        evidence_json = json.dumps(specialist_outputs, indent=2)
        prompt = _FUSION_PROMPT.format(
            evidence_json=evidence_json,
            threshold=CONFIDENCE_THRESHOLD,
        )
        raw = call_gemini_vision(
            prompt=prompt,
            screenshot=state["screenshot"],
            api_key=GEMINI_API_KEY
        )
        log_event(f"fusion result:\n{raw}")
        result = parse_json_response(raw)
        return _build_state_update(result, state, iteration, forced=False)

    except Exception as exc:
        return _error_fallback(state, iteration, str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_state_update(result: dict, state: GeoState, iteration: int, forced: bool) -> dict:
    """Convert Gemini fusion output into a GeoState update."""
    belief_state = result.get("belief_state", [])
    action_raw = result.get("action", {})

    # Ensure belief_state is sorted by confidence descending
    belief_state = sorted(belief_state, key=lambda x: x.get("confidence", 0), reverse=True)

    # Normalise action
    action = _normalise_action(action_raw, belief_state)

    final_guess = {}
    if action["type"] == "GUESS":
        final_guess = {
            "lat": action.get("lat", 0.0),
            "lon": action.get("lon", 0.0),
            "country": belief_state[0].get("country") if belief_state else None,
            "confidence": belief_state[0].get("confidence", 0.0) if belief_state else 0.0,
        }
    max_iter = int(state.get("max_iterations", 0))
    next_iteration = min(iteration + 1, max_iter)

    return {
        "belief_state": belief_state,
        "action": action,
        "final_guess": final_guess,
        "iteration": next_iteration,
        "error": '',
    }


def _normalise_action(action_raw: dict, belief_state: list) -> dict:
    """Ensure action has correct structure regardless of model output quirks."""
    action_type = str(action_raw.get("type", "GUESS")).upper()

    if action_type == "GUESS":
        # Fall back to top belief state coordinates if model omitted them
        top = belief_state[0] if belief_state else {}
        return {
            "type": "GUESS",
            "lat": action_raw.get("lat", top.get("lat", 0.0)),
            "lon": action_raw.get("lon", top.get("lon", 0.0)),
        }
    if action_type == "ROTATE":
        return {"type": "ROTATE", "degrees": int(action_raw.get("degrees", 90))}
    if action_type == "MOVE":
        return {"type": "MOVE", "direction": action_raw.get("direction", "forward")}

    # Unknown action — default to GUESS
    top = belief_state[0] if belief_state else {}
    return {"type": "GUESS", "lat": top.get("lat", 0.0), "lon": top.get("lon", 0.0)}


def _forced_guess(state: GeoState, specialist_outputs: dict, iteration: int) -> dict:
    """Called when the iteration budget is exhausted — commit the best guess we have."""
    # Use existing belief state if available, otherwise create a minimal one
    existing_belief = state.get("belief_state", [])
    if existing_belief:
        top = existing_belief[0]
    else:
        top = {"country": "Unknown", "lat": 0.0, "lon": 0.0, "confidence": 0.0}

    action = {"type": "GUESS", "lat": top.get("lat", 0.0), "lon": top.get("lon", 0.0)}
    final_guess = {
        "lat": top.get("lat", 0.0),
        "lon": top.get("lon", 0.0),
        "country": top.get("country"),
        "confidence": top.get("confidence", 0.0),
        "forced": True,
    }
    max_iter = int(state.get("max_iterations", 0))
    next_iteration = min(iteration + 1, max_iter)

    return {
        "belief_state": existing_belief,
        "action": action,
        "final_guess": final_guess,
        "iteration": next_iteration,
        "error": '',
    }


def _error_fallback(state: GeoState, iteration: int, error_msg: str) -> dict:
    """Return a safe fallback state when fusion fails."""
    action = {"type": "GUESS", "lat": 0.0, "lon": 0.0}
    max_iter = int(state.get("max_iterations", 0))
    next_iteration = min(iteration + 1, max_iter)

    return {
        "belief_state": [],
        "action": action,
        "final_guess": {"lat": 0.0, "lon": 0.0, "country": None, "confidence": 0.0},
        "iteration": next_iteration,
        "error": f"Fusion failed: {error_msg}",
    }
