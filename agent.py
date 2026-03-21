import json
from pathlib import Path
from dataclasses import dataclass
from http.client import HTTPException
from typing import Any

from game import Game
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
from graphs.state import GeoState
from util import log_event


@dataclass
class AnalysisResult:
    belief_state: list
    action: dict
    final_guess: dict
    specialist_outputs: dict
    error: str


class Agent:
    def __init__(self, game: Game = None):
        self.game = game
        self.frame: str = ''
        self.belief_state: list[dict[str, Any]] = []
        self.action_history: list[dict[str, Any]] = []
        self.geo_graph = None

        self.initialize_graph()

    def initialize_graph(self):
        graph = StateGraph(GeoState)

        graph.add_node("ingest", ingest_node)
        graph.add_node("text_language", text_language_node)
        graph.add_node("architecture", architecture_node)
        graph.add_node("climate_terrain", climate_terrain_node)
        graph.add_node("vegetation", vegetation_node)
        graph.add_node("road_infra", road_infra_node)
        graph.add_node("fusion_planner", fusion_planner_node)

        graph.set_entry_point("ingest")

        # ingest → all specialists in parallel (fan-out)
        for specialist in ["text_language", "architecture", "climate_terrain", "vegetation", "road_infra"]:
            graph.add_edge("ingest", specialist)

        # all specialists → fusion_planner (fan-in)
        for specialist in ["text_language", "architecture", "climate_terrain", "vegetation", "road_infra"]:
            graph.add_edge(specialist, "fusion_planner")

        graph.add_edge("fusion_planner", END)

        self.geo_graph = graph.compile()


    def analyze(self, new_frame: str, heading: float, max_iter: int, cur_iter: int) -> AnalysisResult:
        # TODO incorporate heading information (to deduce sun direction)
        initial_state: GeoState = {
            "screenshot": new_frame,
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
            error=result.get("error", '')
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
        for i in range(max_iter):
            frame: str = self.game.render_image()
            result: AnalysisResult = self.analyze(frame, self.game.heading, max_iter, i)
            if len(result.error) > 0:
                return {"error": result.error}
            action: str = result.action["type"]
            if action == "GUESS":
                guess_dist: float = self.game.guess(result.final_guess["lat"], result.final_guess["lon"])
                log_event(f"GUESS RESULT: {guess_dist}km")
                return {
                    "final_guess": result.final_guess,
                    "belief_state": result.belief_state,
                    "iterations_used": i + 1,
                    "errors": result.error
                }
            if action == "ROTATE":
                self.game.turn(result.action["degrees"], 0)
            elif action == "MOVE":
                self.game.move_forward()
        return {"error": f"Agent did not give a guess after {max_iter} iterations"}
    
    def export_geo_graph_image(self, output_path: str = "geo_graph.png") -> str:
        """
        Export a visualization of ``geo_graph``.

        Preferred output is PNG. If PNG rendering fails in the current environment,
        a Mermaid diagram is saved next to the requested output path and a clear
        RuntimeError is raised with the fallback path.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        drawable = self.geo_graph.get_graph()

        # Primary path: PNG bytes from Mermaid rendering.
        if out_path.suffix.lower() == ".png":
            try:
                png_bytes = drawable.draw_mermaid_png()
                out_path.write_bytes(png_bytes)
                return str(out_path)
            except Exception as exc:
                # Fallback: persist Mermaid source so users can still render/view it.
                mermaid_path = out_path.with_suffix(".mmd")
                mermaid_path.write_text(drawable.draw_mermaid(), encoding="utf-8")
                raise RuntimeError(
                    f"PNG export failed ({exc}). Mermaid diagram saved to {mermaid_path}."
                ) from exc

        # Non-PNG outputs are written as Mermaid diagram text.
        out_path.write_text(drawable.draw_mermaid(), encoding="utf-8")
        return str(out_path)

if __name__ == "__main__":
    game: Game = Game()
    game.set_to_random_street_view()
    agent = Agent(game)
    agent.export_geo_graph_image("geo_graph_old.png")
    print(agent.run(max_iter=3))
