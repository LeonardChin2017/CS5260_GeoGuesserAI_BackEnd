"""
Background asyncio runner that drives the LangGraph pipeline and updates
the shared AGENT_STATE dict that the frontend polls via /api/agent/status
and /api/agent/frame.

Step mapping (old 8 steps → new 4 steps visible to the frontend):
  capture  → "Capturing Street View frame"
  analyze  → "Analysing visual clues"
  reason   → "Fusing evidence & planning"
  guess    → "Placing guess"
"""
import asyncio
import os
from typing import Any, Dict, Optional

from graphs.geoguessr_graph import geo_graph
from graphs.state import GeoState
from graphs.action_executor import GameView, execute_action, fetch_streetview_screenshot

_PIPELINE_STEPS = [
    ("capture", "Capturing Street View frame"),
    ("analyze", "Analysing visual clues"),
    ("reason",  "Fusing evidence & planning"),
    ("guess",   "Placing guess"),
]


async def run_langgraph_agent(
    agent_state: Dict[str, Any],
    agent_lock: asyncio.Lock,
    start_lat: float,
    start_lon: float,
    start_heading: float,
    max_iterations: int,
) -> None:
    """
    Background coroutine.  Runs the LangGraph pipeline, updating
    agent_state in place so the polling endpoints stay current.
    """

    def _set_steps(statuses: Dict[str, str]) -> None:
        steps = []
        for sid, msg in _PIPELINE_STEPS:
            steps.append({"id": sid, "message": msg, "status": statuses.get(sid, "pending")})
        agent_state["steps"] = steps

    def _update_game_from_result(result: Dict[str, Any], view: GameView) -> None:
        """Push LangGraph result fields back into agent_state["game"]."""
        game = dict(agent_state.get("game") or {})
        # update view position from current GameView
        game["view_lat"] = view.lat
        game["view_lon"] = view.lon
        game["heading"] = view.heading

        belief = result.get("belief_state") or []
        if belief:
            top = belief[0]
            game["confidence"] = float(top.get("confidence", 0.0))
            game["best_country_guess"] = top.get("country")
            game["guess_lat"] = float(top.get("lat", game.get("guess_lat", 0.0)))
            game["guess_lon"] = float(top.get("lon", game.get("guess_lon", 0.0)))
            game["detected_clues"] = [top.get("evidence", "")]

        final = result.get("final_guess")
        if final:
            game["guess_lat"] = float(final.get("lat", game.get("guess_lat", 0.0)))
            game["guess_lon"] = float(final.get("lon", game.get("guess_lon", 0.0)))
            game["confidence"] = float(final.get("confidence", game.get("confidence", 0.0)))

        agent_state["game"] = game

    maps_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    view = GameView(start_lat, start_lon, start_heading)
    belief_state: list = []
    action_history: list = []

    async with agent_lock:
        _set_steps({"capture": "running"})

    try:
        for iteration in range(max_iterations):
            async with agent_lock:
                if agent_state.get("stop"):
                    break

            # --- capture ---
            async with agent_lock:
                _set_steps({"capture": "running", "analyze": "pending", "reason": "pending", "guess": "pending"})

            # Fetch Street View screenshot (blocking IO → run in executor)
            loop = asyncio.get_event_loop()
            try:
                screenshot = await loop.run_in_executor(
                    None,
                    lambda v=view, k=maps_key: fetch_streetview_screenshot(v, k),
                )
            except Exception as exc:
                async with agent_lock:
                    agent_state["error"] = f"Street View fetch failed: {exc}"
                    agent_state["running"] = False
                return

            # Store frame in agent_state for /api/agent/frame polling
            async with agent_lock:
                # screenshot is "data:image/jpeg;base64,..."
                _b64 = screenshot.split(",", 1)[1] if "," in screenshot else screenshot
                agent_state["frame"] = _b64
                agent_state["frame_mime"] = "image/jpeg"
                from datetime import datetime
                agent_state["last_frame_at"] = datetime.utcnow().isoformat()
                _set_steps({"capture": "done", "analyze": "running", "reason": "pending", "guess": "pending"})

            # --- analyze + reason (run graph) ---
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
                result = await loop.run_in_executor(None, lambda s=geo_state: geo_graph.invoke(s))
            except Exception as exc:
                async with agent_lock:
                    agent_state["error"] = f"Graph error: {exc}"
                    agent_state["running"] = False
                return

            belief_state = result.get("belief_state") or []
            action_history = result.get("action_history") or []
            action = result.get("action") or {"type": "GUESS"}

            async with agent_lock:
                _set_steps({"capture": "done", "analyze": "done", "reason": "done", "guess": "running"})
                _update_game_from_result(result, view)

            # --- guess or explore ---
            if action.get("type") == "GUESS":
                async with agent_lock:
                    game = dict(agent_state.get("game") or {})
                    final = result.get("final_guess") or {}
                    if final:
                        game["guess_lat"] = float(final.get("lat", game.get("guess_lat", 0.0)))
                        game["guess_lon"] = float(final.get("lon", game.get("guess_lon", 0.0)))
                    # score the guess
                    from app import _distance_km, _score_from_distance
                    dist = _distance_km(
                        float(game.get("guess_lat", 0.0)),
                        float(game.get("guess_lon", 0.0)),
                        float(game.get("target_lat", 0.0)),
                        float(game.get("target_lon", 0.0)),
                    )
                    game["final_distance_km"] = dist
                    game["score"] = _score_from_distance(dist)
                    agent_state["game"] = game
                    _set_steps({"capture": "done", "analyze": "done", "reason": "done", "guess": "done"})
                    agent_state["running"] = False
                break
            else:
                # ROTATE or MOVE — execute and loop
                async with agent_lock:
                    _set_steps({"capture": "done", "analyze": "done", "reason": "done", "guess": "pending"})

                if not maps_key:
                    # No Street View key — can't physically move, so just guess
                    async with agent_lock:
                        agent_state["running"] = False
                    break

                try:
                    new_screenshot, new_view = await loop.run_in_executor(
                        None,
                        lambda a=action, v=view, k=maps_key: execute_action(a, v, k),
                    )
                    view = new_view
                except Exception:
                    # Can't execute action — commit a guess with what we have
                    async with agent_lock:
                        agent_state["running"] = False
                    break

    except Exception as exc:
        async with agent_lock:
            agent_state["error"] = f"Runner error: {exc}"
            agent_state["running"] = False
        return

    async with agent_lock:
        agent_state["running"] = False
