"""
Background asyncio runner that drives the LangGraph pipeline and updates
the shared AGENT_STATE dict that the frontend polls via /api/agent/status
and /api/agent/frame.

Step mapping visible to the frontend (8 steps):
  capture       → "Capturing Street View frame"
  text_language → "Text & Language"
  architecture  → "Architecture"
  climate_terrain → "Climate & Terrain"
  vegetation    → "Vegetation"
  road_infra    → "Road & Infrastructure"
  reason        → "Fusing evidence & planning"
  guess         → "Placing guess"
"""
import asyncio
import math
import os
import time
from datetime import datetime, UTC
from typing import Any, Dict

from app import AgentState, GameState
from graphs.action_executor import GameView, execute_action, fetch_streetview_screenshot
from graphs.geoguessr_graph import geo_graph
from graphs.state import GeoState
from util import _0_if_nan

_PIPELINE_STEPS = [
    ("capture",        "Capturing Street View frame"),
    ("text_language",  "Text & Language"),
    ("architecture",   "Architecture"),
    ("climate_terrain","Climate & Terrain"),
    ("vegetation",     "Vegetation"),
    ("road_infra",     "Road & Infrastructure"),
    ("reason",         "Fusing evidence & planning"),
    ("guess",          "Placing guess"),
]

_SPECIALIST_IDS = ["text_language", "architecture", "climate_terrain", "vegetation", "road_infra"]


# ---------------------------------------------------------------------------
# Inline scoring (avoids cross-module import inside async lock)
# ---------------------------------------------------------------------------

def _calc_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    lon1r, lon2r = math.radians(lon1), math.radians(lon2)
    x = (lon2r - lon1r) * max(0.1, abs((lat1r + lat2r) / 2.0))
    y = lat2r - lat1r
    return math.sqrt(x * x + y * y) * 6371.0


def _calc_score(dist_km: float) -> int:
    return int(max(0.0, 5000.0 - dist_km * 4.0))


# ---------------------------------------------------------------------------
# Extract a short human-readable detail from a specialist output dict
# ---------------------------------------------------------------------------

def _specialist_detail(key: str, output: Dict[str, Any]) -> str:
    conf = output.get("confidence", 0.0)
    conf_pct = f"{int(conf * 100)}%"
    evidence = str(output.get("evidence", ""))[:80]
    if key == "text_language":
        scripts = output.get("detected_scripts") or []
        hints = output.get("language_hints") or []
        tag = ", ".join(scripts[:2] + hints[:2]) or evidence
    elif key == "architecture":
        styles = output.get("building_styles") or []
        tag = ", ".join(styles[:2]) or evidence
    elif key == "climate_terrain":
        zone = output.get("climate_zone", "")
        terrain = output.get("terrain_type", "")
        tag = f"{zone} / {terrain}" if zone or terrain else evidence
    elif key == "vegetation":
        biome = output.get("biome", "")
        vtype = output.get("vegetation_type", "")
        tag = biome or vtype or evidence
    elif key == "road_infra":
        side = output.get("driving_side", "")
        tag = f"{side}-hand traffic" if side and side != "unknown" else evidence
    else:
        tag = evidence
    return f"{tag} ({conf_pct})" if tag else conf_pct


async def run_langgraph_agent(
    agent_state: AgentState,
    agent_lock: asyncio.Lock,
    start_lat: float,
    start_lon: float,
    start_heading: float,
    max_iterations: int,
) -> None:
    """
    Background coroutine. Runs the LangGraph pipeline, updating
    agent_state in place so the polling endpoints stay current.
    """

    def _set_steps(statuses: Dict[str, str], details: Dict[str, str] | None = None) -> None:
        steps = []
        for sid, msg in _PIPELINE_STEPS:
            step: Dict[str, Any] = {"id": sid, "message": msg, "status": statuses.get(sid, "pending")}
            if details and sid in details:
                step["detail"] = details[sid]
            steps.append(step)
        agent_state.steps = steps

    def _update_game_from_result(result: Dict[str, Any], view: GameView) -> None:
        """Push LangGraph result fields back into agent_state.game."""
        game: GameState = GameState() if agent_state.game is None else agent_state.game
        game.view_lat = view.lat
        game.view_lon = view.lon
        game.heading = view.heading

        belief = result.get("belief_state") or []
        if belief:
            top = belief[0]
            game.confidence = float(top.get("confidence", 0.0))
            game.best_country_guess = top.get("country")
            game.guess_lat = float(top.get("lat", _0_if_nan(game.guess_lat)))
            game.guess_lon = float(top.get("lon", _0_if_nan(game.guess_lon)))
            game.detected_clues = [top.get("evidence", "")]

        final = result.get("final_guess")
        if final:
            game.guess_lat = float(final.get("lat", _0_if_nan(game.guess_lat)))
            game.guess_lon = float(final.get("lon", _0_if_nan(game.guess_lon)))
            game.confidence = float(final.get("confidence", _0_if_nan(game.confidence)))

        agent_state.game = game

    maps_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()

    view = GameView(start_lat, start_lon, start_heading)
    belief_state: list = []
    action_history: list = []

    async with agent_lock:
        _set_steps({"capture": "running"})

    try:
        for iteration in range(max_iterations):
            async with agent_lock:
                if agent_state.stop:
                    break

            # ----------------------------------------------------------------
            # 1. CAPTURE — fetch Street View frame
            # ----------------------------------------------------------------
            async with agent_lock:
                _set_steps({"capture": "running"})

            loop = asyncio.get_event_loop()
            try:
                screenshot = await loop.run_in_executor(
                    None,
                    lambda v=view, k=maps_key: fetch_streetview_screenshot(v, k),
                )
            except Exception as exc:
                async with agent_lock:
                    agent_state.error = f"Street View fetch failed: {exc}"
                    agent_state.running = False
                return

            _b64 = screenshot.split(",", 1)[1] if "," in screenshot else screenshot
            image_id = f"cap-{int(time.time() * 1000)}"

            async with agent_lock:
                agent_state.frame = _b64
                agent_state.frame_mime = "image/jpeg"
                agent_state.last_frame_at = datetime.now(UTC).isoformat()
                # Populate captured_images so the UI thumbnail shows the frame
                agent_state.captured_images = [{
                    "id": image_id,
                    "captured_at": datetime.now(UTC).isoformat(),
                    "mime": "image/jpeg",
                    "data": _b64,
                    "screenshot_url": None,
                    "saved_url": None,
                }]
                # Show all specialists as running while the graph executes
                _set_steps(
                    {"capture": "done",
                     "text_language": "running", "architecture": "running",
                     "climate_terrain": "running", "vegetation": "running",
                     "road_infra": "running",
                     "reason": "pending", "guess": "pending"},
                )

            # ----------------------------------------------------------------
            # 2. ANALYSE — run LangGraph (5 specialists in parallel + fusion)
            # ----------------------------------------------------------------
            geo_state = GeoState(
                screenshot=screenshot,
                iteration=iteration,
                max_iterations=max_iterations,
                specialist_outputs={},
                belief_state=belief_state,
                action={"type": "GUESS"},
                action_history=action_history,
                final_guess=None,
                error=None,
            )

            try:
                result = await loop.run_in_executor(
                    None, lambda s=geo_state: geo_graph.invoke(s)
                )
            except Exception as exc:
                async with agent_lock:
                    agent_state.error = f"Graph error: {exc}"
                    agent_state.running = False
                return

            belief_state = result.get("belief_state") or []
            action_history = result.get("action_history") or []
            action = result.get("action") or {"type": "GUESS"}
            specialist_outputs = result.get("specialist_outputs") or {}

            # Build per-specialist detail strings for the step panel
            spec_details: Dict[str, str] = {}
            for sid in _SPECIALIST_IDS:
                out = specialist_outputs.get(sid)
                if out and "error" not in out:
                    spec_details[sid] = _specialist_detail(sid, out)
                elif out:
                    spec_details[sid] = f"Error: {out.get('error', 'unknown')}"

            # Build fusion/reason detail
            if belief_state:
                top = belief_state[0]
                reason_detail = (
                    f"{top.get('country', '?')} "
                    f"({int(float(top.get('confidence', 0)) * 100)}%) — "
                    f"{str(top.get('evidence', ''))[:60]}"
                )
            else:
                reason_detail = "No belief state"

            async with agent_lock:
                _update_game_from_result(result, view)
                agent_state.specialist_outputs = specialist_outputs # TODO fix
                _set_steps(
                    {"capture": "done",
                     "text_language": "done", "architecture": "done",
                     "climate_terrain": "done", "vegetation": "done",
                     "road_infra": "done",
                     "reason": "done", "guess": "running"},
                    details={**spec_details, "reason": reason_detail},
                )

            # ----------------------------------------------------------------
            # 3. GUESS or EXPLORE
            # ----------------------------------------------------------------
            if action.get("type") == "GUESS":
                async with agent_lock:
                    game: GameState = GameState() if agent_state.game is None else agent_state.game
                    final = result.get("final_guess") or {}
                    if final:
                        game.guess_lat = float(final.get("lat", _0_if_nan(game.guess_lat)))
                        game.guess_lon = float(final.get("lon", _0_if_nan(game.guess_lon)))

                    dist = _calc_distance_km(
                        _0_if_nan(game.guess_lat),
                        _0_if_nan(game.guess_lon),
                        _0_if_nan(game.target_lat),
                        _0_if_nan(game.target_lon)
                    )
                    game.final_distance_km = dist
                    game.score = _calc_score(dist)
                    agent_state.game = game

                    guess_detail = (
                        f"Distance: {dist:.0f} km | Score: {game.score}"
                    )
                    _set_steps(
                        {"capture": "done",
                         "text_language": "done", "architecture": "done",
                         "climate_terrain": "done", "vegetation": "done",
                         "road_infra": "done",
                         "reason": "done", "guess": "done"},
                        details={**spec_details, "reason": reason_detail, "guess": guess_detail},
                    )
                    agent_state.running = False
                break

            else:
                # ROTATE or MOVE — continue exploring
                async with agent_lock:
                    _set_steps(
                        {"capture": "done",
                         "text_language": "done", "architecture": "done",
                         "climate_terrain": "done", "vegetation": "done",
                         "road_infra": "done",
                         "reason": "done", "guess": "pending"},
                        details={**spec_details, "reason": reason_detail},
                    )

                if not maps_key:
                    async with agent_lock:
                        agent_state.running = False
                    break

                try:
                    _, new_view = await loop.run_in_executor(
                        None,
                        lambda a=action, v=view, k=maps_key: execute_action(a, v, k),
                    )
                    view = new_view
                except Exception:
                    async with agent_lock:
                        agent_state.running = False
                    break

    except Exception as exc:
        async with agent_lock:
            agent_state.error = f"Runner error: {exc}"
            agent_state.running = False
        return

    async with agent_lock:
        agent_state.running = False
