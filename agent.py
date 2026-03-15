from dataclasses import dataclass
from http.client import HTTPException
from typing import Any

from game import Game
from graphs.geoguessr_graph import geo_graph
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
    def __init__(self):
        self.frame: str = ''
        self.belief_state: list[dict[str, Any]] = []
        self.action_history: list[dict[str, Any]] = []

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
            result = geo_graph.invoke(initial_state)
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

    def run(self, game: Game, max_iter: int) -> dict[str, Any]:
        for i in range(max_iter):
            frame: str = game.render_image()
            result: AnalysisResult = self.analyze(frame, game.heading, max_iter, i)
            if len(result.error) > 0:
                return {"error": result.error}
            action: str = result.action["type"]
            if action == "GUESS":
                guess_dist: float = game.guess(result.final_guess["lat"], result.final_guess["lon"])
                log_event(f"GUESS RESULT: {guess_dist}km")
                return {
                    "final_guess": result.final_guess,
                    "belief_state": result.belief_state,
                    "iterations_used": i + 1,
                    "errors": result.error
                }
            if action == "ROTATE":
                game.turn(result.action["degrees"], 0)
            elif action == "MOVE":
                game.move_forward()
        return {"error": f"Agent did not give a guess after {max_iter} iterations"}


if __name__ == "__main__":
    game: Game = Game()
    game.set_to_random_street_view()
    print(Agent().run(game, max_iter=3))
