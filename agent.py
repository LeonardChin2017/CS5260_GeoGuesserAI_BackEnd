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
        # Reset transient fields each cycle so specialists always write fresh output.
        return {
            "screenshot": image,
            "specialist_outputs": {},
            "action": {},
            "final_guess": {},
            "error": '',
        }

    def initialize_graph(self):
        graph = StateGraph(GeoState)

        graph.add_node("render_image", self.render_image)
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

        graph.set_entry_point("render_image")

        # render_image → ingest
        graph.add_edge("render_image", "ingest")

        # ingest → all specialists in parallel (fan-out)
        for specialist in ["text_language", "architecture", "climate_terrain", "vegetation", "road_infra"]:
            graph.add_edge("ingest", specialist)

        # all specialists → fusion_planner (fan-in)
        for specialist in ["text_language", "architecture", "climate_terrain", "vegetation", "road_infra"]:
            graph.add_edge(specialist, "fusion_planner")

        graph.add_conditional_edges(
            "fusion_planner",
            self.route_action,
            {
                "GUESS": "execute_guess",
                "ROTATE": "execute_rotate",
                "MOVE": "execute_move",
            },
        )
        graph.add_edge("execute_guess", END)
        graph.add_conditional_edges(
            "execute_rotate",
            self.route_exploration_loop,
            {
                "CONTINUE": "render_image",
                "STOP": END,
            },
        )
        graph.add_conditional_edges(
            "execute_move",
            self.route_exploration_loop,
            {
                "CONTINUE": "render_image",
                "STOP": END,
            },
        )

        self.geo_graph = graph.compile()

    @staticmethod
    def route_action(state: GeoState) -> str:
        action_type = str(state.get("action", {}).get("type", "GUESS")).upper()
        if action_type in {"GUESS", "ROTATE", "MOVE"}:
            return action_type
        return "GUESS"

    @staticmethod
    def route_exploration_loop(state: GeoState) -> str:
        iteration = int(state.get("iteration", 0))
        max_iterations = int(state.get("max_iterations", 0))
        return "CONTINUE" if iteration < max_iterations else "STOP"

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
        return {"final_guess": final_guess}

    def execute_rotate(self, state: GeoState) -> dict[str, Any]:
        if self.game is None:
            raise ValueError("Game instance is not set in Agent.")
        degrees = float(state.get("action", {}).get("degrees", 90.0))
        self.game.turn(degrees, 0)
        return {}

    def execute_move(self, state: GeoState) -> dict[str, Any]:
        if self.game is None:
            raise ValueError("Game instance is not set in Agent.")
        self.game.move_forward()
        return {}


    def analyze(self, max_iter: int, cur_iter: int = 0) -> AnalysisResult:
        # TODO incorporate heading information (to deduce sun direction)
        initial_state: GeoState = {
            "iteration": cur_iter,
            "max_iterations": max_iter,
            "specialist_outputs": {},  # fresh each iteration TODO keep some information across iteration
            "belief_state": self.belief_state,
            "action": {},
            "final_guess": {},
            "error": '',
        }
        try:
            result = self.geo_graph.invoke(initial_state)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        self.belief_state = result.get("belief_state", self.belief_state)
        action: dict[str, Any] = result.get("action", {})
        self.action_history.append(action)
        return AnalysisResult(
            belief_state=self.belief_state,
            action=action,
            final_guess=result.get("final_guess", {}),
            specialist_outputs=result.get("specialist_outputs", {}),
            error=result.get("error", ''),
            iteration=int(result.get("iteration", cur_iter)),
        )

    def stream_analyze(self, new_frame: str, heading: float, max_iter: int, cur_iter: int):
        initial_state: GeoState = {
            "screenshot": new_frame,
            "iteration": cur_iter,
            "max_iterations": max_iter,
            "specialist_outputs": {},
            "belief_state": self.belief_state,
            "action": {},
            "final_guess": {},
            "error": '',
        }
        try:
            for chunk in self.geo_graph.stream(initial_state, stream_mode="updates"):
                # Accumulate state internally
                for node_name, state_update in chunk.items():
                    if "belief_state" in state_update:
                        self.belief_state = state_update["belief_state"]
                    if "action" in state_update:
                        self.action_history.append(state_update["action"])
                # Yield chunk as Server-Sent Event
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    def run(self, max_iter: int) -> dict[str, Any]:
        result: AnalysisResult = self.analyze(max_iter=max_iter, cur_iter=0)
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
    game.set_to_random_street_view()
    agent = Agent(game)
    agent.export_geo_graph_image("geo_graph_new.png")
    print(agent.run(max_iter=3))
