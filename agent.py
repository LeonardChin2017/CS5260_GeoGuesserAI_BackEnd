import asyncio
import base64
from dataclasses import dataclass
from http.client import HTTPException
from typing import Any

from dotenv import load_dotenv

from game import Game
from graphs.geoguessr_graph import geo_graph
from graphs.state import GeoState


@dataclass
class AnalysisResult:
    belief_state: list
    action: dict
    final_guess: dict
    specialist_outputs: dict
    error: str


class Agent:
    def __init__(self):
        self.lock: asyncio.Lock = asyncio.Lock()
        self.running: bool = False
        self.stop: bool = True
        self.frames: list[str] = []
        self.belief_state: list[dict[str, Any]] = []
        self.action_history: list[dict[str, Any]] = []

    def stop(self) -> None:
        self.running = False
        self.stop = True

    def status(self) -> (bool, bool):
        return self.running, self.stop

    def frame(self) -> str:
        return self.frames[-1] if len(self.frames) > 0 else ''

    def analyze(self, new_frame: str, max_iter: int, cur_iter: int) -> AnalysisResult:
        self.frames.append(new_frame)
        initial_state: GeoState = {
            "screenshot": new_frame,
            "iteration": cur_iter,
            "max_iterations": max_iter,
            "specialist_outputs": {},  # fresh each iteration TODO keep some information across iteration
            "belief_state": self.belief_state,
            "action": {},
            "final_guess": None,
            "error": None,
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

    def captures(self) -> list[str]:
        return self.frames


def run(max_iter=4) -> float:
    agent: Agent = Agent()
    game: Game = Game()
    for i in range(max_iter):
        frame: str = base64.b64encode(game.render_image().data).decode("utf-8")
        result: AnalysisResult = agent.analyze(frame, max_iter, i)
        if len(result.error) > 0:
            raise RuntimeError(result.error)
        action: str = result.action["type"]
        if action == "GUESS":
            return game.guess(result.final_guess["lat"], result.final_guess["lon"])
        if action == "ROTATE":
            game.turn(result.action["degree"], 0)
        elif action == "MOVE":
            game.move_forward()
    raise RuntimeError(f"Agent did not give a guess after {max_iter} iterations")


if __name__ == "__main__":
    load_dotenv()
    agent: Agent = Agent()
    game: Game = Game()
    frame: str = base64.b64encode(game.render_image().data).decode("utf-8")
    result: AnalysisResult = agent.analyze(frame, 3, 0)
    print(result)
    if result.action["type"] == "GUESS":
        print(f"{game.guess(result.final_guess["lat"], result.final_guess["lon"])}km")