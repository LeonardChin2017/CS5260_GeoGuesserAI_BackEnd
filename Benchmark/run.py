import json
from pathlib import Path
from sys import stderr
from typing import Optional

from tqdm import tqdm

from agent import Agent
from game import Game
from util import LOCATION_DATABASE

OUTPUT_FILE = Path("Benchmark", "results.json")

if __name__ == '__main__':
    # Load existing results if file exists
    all_results: dict[str, object] = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r") as f:
            all_results = json.load(f)
    try:
        for i, (lat, lng) in enumerate(tqdm(LOCATION_DATABASE)):
            i: str = str(i)
            result: Optional[dict[str, object]] = all_results.get(i)
            if result is not None and len(result.get("error", '')) <= 0:
                continue
            game: Game = Game()
            game.reset(lat, lng)
            agent = Agent(game)
            result: dict[str, object] = agent.run(max_iter=5)
            all_results[i] = result
    except Exception as e:
        print(e, file=stderr)
        pass
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)