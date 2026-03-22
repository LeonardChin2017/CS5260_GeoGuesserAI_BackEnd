import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from game import Game
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
from langgraph.graph import StateGraph, END
from graphs.state import GeoState
from graphs.nodes.ingest import ingest_node
from graphs.nodes.specialists import (
    text_language_node,
    architecture_node,
    climate_terrain_node,
    vegetation_node,
    road_infra_node,
)
from graphs.nodes.fusion import fusion_planner_node
from util import log_event


@dataclass
class AnalysisResult:
    belief_state: list
    action: dict
    final_guess: dict
    specialist_outputs: dict
    error: str
    iteration: int = 0


class Agent:
    def __init__(self, game: Game = None):
        self.game = game
        self.frame: str = ''
        self.belief_state: list[dict[str, Any]] = []
        self.action_history: list[dict[str, Any]] = []
        self.geo_graph = None

        self.initialize_graph()

    def render_image(self, state: GeoState) -> dict[str, Any]:
        if self.game is None:
            raise ValueError("Game instance is not set in Agent.")
        image = self.game.render_image()
        self.frame = image
        return {"screenshot": image, "message": "Agent updated the street view image for the current iteration."}

    def initialize_graph(self):
        graph = StateGraph(GeoState)

        graph.add_node("mode_gate", self.mode_gate)
        graph.add_node("render_image", self.render_image)
        graph.add_node("dispatch_action", self.dispatch_action)
        graph.add_node("ingest", ingest_node)
        graph.add_node("text_language", text_language_node)
        graph.add_node("architecture", architecture_node)
        graph.add_node("climate_terrain", climate_terrain_node)
        graph.add_node("vegetation", vegetation_node)
        graph.add_node("road_infra", road_infra_node)
        graph.add_node("fusion_planner", fusion_planner_node)
        graph.add_node("execute_guess", self.execute_guess)
        graph.add_node("execute_rotate", self.execute_rotate)
        graph.add_node("execute_move", self.execute_move)
        graph.add_node("iteration_guard", self.iteration_guard)

        graph.set_entry_point("mode_gate")
        graph.add_conditional_edges(
            "mode_gate",
            self.route_mode,
            {
                "RUN": "render_image",
                "ANALYZE": "ingest",
            },
        )

        # run mode: render_image → ingest
        graph.add_edge("render_image", "ingest")

        # ingest → all specialists in parallel (fan-out)
        for specialist in ["text_language", "architecture", "climate_terrain", "vegetation", "road_infra"]:
            graph.add_edge("ingest", specialist)

        # all specialists → fusion_planner (fan-in)
        for specialist in ["text_language", "architecture", "climate_terrain", "vegetation", "road_infra"]:
            graph.add_edge(specialist, "fusion_planner")

        graph.add_conditional_edges(
            "fusion_planner",
            self.route_after_fusion,
            {
                "STOP": END,
                "ACTION": "dispatch_action",
            },
        )
        graph.add_conditional_edges(
            "dispatch_action",
            self.route_action,
            {
                "GUESS": "execute_guess",
                "ROTATE": "execute_rotate",
                "MOVE": "execute_move",
            },
        )
        graph.add_edge("execute_guess", END)
        graph.add_edge("execute_rotate", "iteration_guard")
        graph.add_edge("execute_move", "iteration_guard")
        graph.add_conditional_edges(
            "iteration_guard",
            self.route_exploration_loop,
            {
                "CONTINUE": "render_image",
                "STOP": "execute_guess",
            },
        )

        self.geo_graph = graph.compile()

    @staticmethod
    def mode_gate(state: GeoState) -> dict[str, Any]:
        """No-op node used as a conditional entry gate by execution mode."""
        return {"mode": state.get("mode", "run"), "message": f"Agent is starting in {state.get('mode', 'run').upper()} mode."}

    @staticmethod
    def dispatch_action(state: GeoState) -> dict[str, Any]:
        """No-op node to separate mode routing from action routing."""
        return {"action": state.get("action", {}), "message": "Agent is dispatching an action."}

    @staticmethod
    def route_mode(state: GeoState) -> str:
        mode = str(state.get("mode", "run")).lower()
        return "ANALYZE" if mode == "analyze" else "RUN"

    @staticmethod
    def route_after_fusion(state: GeoState) -> str:
        # Analyze mode runs a single screenshot pass and returns immediately.
        mode = str(state.get("mode", "run")).lower()
        return "STOP" if mode == "analyze" else "ACTION"

    @staticmethod
    def route_action(state: GeoState) -> str:
        action_type = str(state.get("action", {}).get("type", "GUESS")).upper()
        if action_type in {"GUESS", "ROTATE", "MOVE"}:
            return action_type
        return "GUESS"

    @staticmethod
    def route_exploration_loop(state: GeoState) -> str:
        decision = str(state.get("loop_decision", "STOP")).upper()
        return "CONTINUE" if decision == "CONTINUE" else "STOP"

    @staticmethod
    def iteration_guard(state: GeoState) -> dict[str, Any]:
        """
        Explicitly validate/check loop counters before deciding whether to continue.
        This makes iteration budget checks visible as a dedicated graph node.
        """
        try:
            iteration = int(state.get("iteration", 0))
        except Exception:
            iteration = 0

        try:
            max_iterations = int(state.get("max_iterations", 0))
        except Exception:
            max_iterations = 0

        iteration = max(0, iteration)
        max_iterations = max(0, max_iterations)

        if iteration >= max_iterations:
            return {"iteration": iteration, "max_iterations": max_iterations, "loop_decision": "STOP", "message": f"Agent has reached the maximum iteration budget ({iteration}/{max_iterations}) and will make a final guess."}
        else:
            return {"iteration": iteration + 1, "max_iterations": max_iterations, "loop_decision": "CONTINUE", "message": f"Agent is continuing to explore (iteration {iteration + 1}/{max_iterations})."}

    def execute_guess(self, state: GeoState) -> dict[str, Any]:
        if self.game is None:
            raise ValueError("Game instance is not set in Agent.")

        final_guess = dict(state.get("final_guess", {}))
        action = state.get("action", {})

        lat = final_guess.get("lat", action.get("lat"))
        lon = final_guess.get("lon", action.get("lon"))
        if lat is None or lon is None:
            return {"error": "Missing GUESS coordinates in action/final_guess."}

        guess_dist: float = self.game.guess(float(lat), float(lon))
        final_guess["distance_km"] = guess_dist
        return {"final_guess": final_guess, "message": f"Agent has decided to make a GUESS at ({lat}, {lon}) with distance {guess_dist:.2f} km."}

    def execute_rotate(self, state: GeoState) -> dict[str, Any]:
        if self.game is None:
            raise ValueError("Game instance is not set in Agent.")
        degrees = float(state.get("action", {}).get("degrees", 90.0))
        self.game.turn(degrees, 0)
        return {"message": f"Agent has decided to rotate the street view by {degrees} degrees."}

    def execute_move(self, state: GeoState) -> dict[str, Any]:
        if self.game is None:
            raise ValueError("Game instance is not set in Agent.")
        self.game.move_forward()
        return {"message": "Agent has decided to move the street view forward."}

    def analyze(self, frame: str, heading: float = 0.0, max_iter: int = 1, cur_iter: int = 0) -> AnalysisResult:
        # Heading is currently unused in graph state but kept for API compatibility.
        _ = heading

        screenshot = frame or ""
        if screenshot.startswith("data:"):
            try:
                screenshot = screenshot.split(",", 1)[1]
            except Exception:
                screenshot = ""

        initial_state: GeoState = {
            "mode": "analyze",
            "screenshot": screenshot,
            "iteration": int(cur_iter),
            "max_iterations": int(max_iter),
            "specialist_outputs": {},
            "belief_state": self.belief_state,
            "action": {},
            "final_guess": {},
            "error": '',
        }

        try:
            raw_result = self.geo_graph.invoke(initial_state)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        self.belief_state = raw_result.get("belief_state", self.belief_state)
        action: dict[str, Any] = raw_result.get("action", {})
        self.action_history.append(action)

        return AnalysisResult(
            belief_state=self.belief_state,
            action=action,
            final_guess=raw_result.get("final_guess", {}),
            specialist_outputs=raw_result.get("specialist_outputs", {}),
            error=raw_result.get("error", ''),
            iteration=int(raw_result.get("iteration", cur_iter)),
        )

    def stream_analyze(self, frame: str, heading: float = 0.0, max_iter: int = 1, cur_iter: int = 0):
        """Yield SSE-formatted updates for one screenshot analysis pass."""
        _ = heading

        screenshot = frame or ""
        if screenshot.startswith("data:"):
            try:
                screenshot = screenshot.split(",", 1)[1]
            except Exception:
                screenshot = ""

        initial_state: GeoState = {
            "mode": "analyze",
            "screenshot": screenshot,
            "iteration": int(cur_iter),
            "max_iterations": int(max_iter),
            "specialist_outputs": {},
            "belief_state": self.belief_state,
            "action": {},
            "final_guess": {},
            "error": '',
        }

        try:
            for event in self.geo_graph.stream(initial_state, stream_mode="updates"):
                log_event(f"stream_analyze event: {event}")
                yield f"data: {json.dumps(event)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "event: done\ndata: {}\n\n"

    def run(self, max_iter: int) -> dict[str, Any]:
        # TODO incorporate heading information (to deduce sun direction)
        initial_state: GeoState = {
            "mode": "run",
            "iteration": 0,
            "max_iterations": max_iter,
            "specialist_outputs": {},
            "belief_state": self.belief_state,
            "action": {},
            "final_guess": {},
            "error": '',
        }
        try:
            raw_result = self.geo_graph.invoke(initial_state)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        self.belief_state = raw_result.get("belief_state", self.belief_state)
        action: dict[str, Any] = raw_result.get("action", {})
        self.action_history.append(action)

        result = AnalysisResult(
            belief_state=self.belief_state,
            action=action,
            final_guess=raw_result.get("final_guess", {}),
            specialist_outputs=raw_result.get("specialist_outputs", {}),
            error=raw_result.get("error", ''),
            iteration=int(raw_result.get("iteration", 0)),
        )

        if len(result.error) > 0:
            return {"error": result.error}
        if len(result.final_guess) > 0:
            distance_km = result.final_guess.get("distance_km")
            if distance_km is not None:
                log_event(f"GUESS RESULT: {distance_km}km")
            return {
                "final_guess": result.final_guess,
                "belief_state": result.belief_state,
                "iterations_used": result.iteration,
                "errors": result.error,
            }
        return {"error": f"Agent did not give a guess after {max_iter} iterations"}
    
    def stream_run(self, max_iter: int):
        """Yield SSE-formatted updates for a full run-mode graph execution."""
        initial_state: GeoState = {
            "mode": "run",
            "iteration": 0,
            "max_iterations": int(max_iter),
            "specialist_outputs": {},
            "belief_state": self.belief_state,
            "action": {},
            "final_guess": {},
            "error": '',
        }

        try:
            for event in self.geo_graph.stream(initial_state, stream_mode="updates"):
                # Keep agent memory in sync while streaming.
                if isinstance(event, dict):
                    for _node_name, state_update in event.items():
                        if not isinstance(state_update, dict):
                            continue
                        if isinstance(state_update.get("belief_state"), list):
                            self.belief_state = state_update["belief_state"]
                        if isinstance(state_update.get("action"), dict):
                            self.action_history.append(state_update["action"])
                log_event(f"stream_run event:\n{json.dumps(event)}")
                yield f"data: {json.dumps(event)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "event: done\ndata: {}\n\n"

    def export_geo_graph_image(self, output_path: str = "geo_graph.png") -> str:
        """
        Export a visualization of ``geo_graph``.

        Preferred output is PNG. Falls back to local browser rendering, then to
        Mermaid source text if image rendering is unavailable.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        drawable = self.geo_graph.get_graph()

        # Primary path: PNG bytes from Mermaid rendering.
        if out_path.suffix.lower() == ".png":
            try:
                png_bytes = drawable.draw_mermaid_png(max_retries=5, retry_delay=2.0)
                out_path.write_bytes(png_bytes)
                return str(out_path)
            except Exception as api_exc:
                # Fallback 1: local browser-based render (no mermaid.ink dependency).
                try:
                    png_bytes = drawable.draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)
                    out_path.write_bytes(png_bytes)
                    return str(out_path)
                except Exception as local_exc:
                    # Fallback 2: persist Mermaid source so execution can continue.
                    mermaid_path = out_path.with_suffix(".mmd")
                    mermaid_path.write_text(drawable.draw_mermaid(), encoding="utf-8")
                    log_event(
                        "[export_geo_graph_image] PNG export failed via API "
                        f"({api_exc}) and PYPPETEER ({local_exc}); wrote {mermaid_path}."
                    )
                    return str(mermaid_path)

        # Non-PNG outputs are written as Mermaid diagram text.
        out_path.write_text(drawable.draw_mermaid(), encoding="utf-8")
        return str(out_path)

if __name__ == "__main__":
    game: Game = Game()
    agent = Agent(game)
    agent.export_geo_graph_image("geo_graph_new.png")

    # Run the full agent loop on a random Street View until it makes a guess or hits max iterations.
    game.set_to_random_street_view()
    print(agent.run(max_iter=3))

    # Analyze a single frame with the agent's current state (without running the full graph).
    game.set_to_random_street_view()
    frame = game.render_image()
    print(agent.analyze(frame=frame))


