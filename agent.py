import asyncio
import base64
import sys
from dataclasses import dataclass
from http.client import HTTPException

from dotenv import load_dotenv

from game import Game
from graphs.geoguessr_graph import geo_graph
from graphs.state import GeoState


@dataclass
class AnalyseResult:
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

    def stop(self) -> None:
        self.running = False
        self.stop = True

    def status(self) -> (bool, bool):
        return self.running, self.stop

    def frame(self) -> str:
        return self.frames[-1] if len(self.frames) > 0 else ''

    def analyze(self, new_frame: str, max_iter: int, cur_iter: int) -> AnalyseResult:
        initial_state: GeoState = {
            "screenshot": new_frame,
            "iteration": cur_iter,
            "max_iterations": max_iter,
            "specialist_outputs": {},
            "belief_state": [],
            "action": {},
            "action_history": [],
            "final_guess": None,
            "error": None,
        }
        try:
            result = geo_graph.invoke(initial_state)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return AnalyseResult(
            belief_state=result.get("belief_state", []),
            action=result.get("action", {}),
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
        result: AnalyseResult = agent.analyze(frame, max_iter, i)
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
    result: AnalyseResult = agent.analyze(frame, 3, 0)
    print(result)
    if result.action["type"] == "GUESS":
        print(f"{game.guess(result.final_guess["lat"], result.final_guess["lon"])}km")