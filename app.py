import asyncio
from dataclasses import asdict
from typing import Any, Optional

from fastapi import FastAPI

from agent import Agent, AnalysisResult
from game import Game

from pydantic import BaseModel

AGENT_LOCK: asyncio.Lock = asyncio.Lock()
AGENT: Optional[Agent] = None

app = FastAPI()


@app.get("/")
def root():
    return {"service": "GGSolver-backend", "message": "API only. Try GET /health."}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/agent/start")
async def start_agent():
    global AGENT
    async with AGENT_LOCK:
        if AGENT is None:
            AGENT = Agent()
    return {"ok": True}


@app.post("/api/agent/stop")
async def stop_agent():
    global AGENT
    AGENT = None
    return {"ok": True}


@app.get("/api/agent/status")
async def agent_status():
    global AGENT
    return {"running": AGENT is not None}


@app.get("/api/agent/frame")
async def agent_frame():
    global AGENT
    out: dict[str, Any] = {}
    async with AGENT_LOCK:
        if AGENT is not None and len(AGENT.frame) > 0:
            out["frame"] = AGENT.frame
    return out


class AnalysisRequest(BaseModel):
    screenshot: str  # base64-encoded image (raw base64)
    heading: float
    max_iter: int
    cur_iter: int


@app.post("/api/agent/analyze")
async def agent_analyze(req: AnalysisRequest):
    """
    Single-iteration: run the LangGraph pipeline on one screenshot.
    Returns belief_state, action, final_guess, specialist_outputs, and errors.
    Useful for debugging individual iterations.
    """
    result: AnalysisResult = AGENT.analyze(req.screenshot, req.heading, req.max_iter, req.cur_iter)
    return asdict(result)


class RunRequest(BaseModel):
    start_lat: float
    start_lon: float
    start_heading: float
    max_iter: int


@app.post("/api/agent/run")
async def agent_run(req: RunRequest):
    """
    Full autopilot loop: runs the multi-iteration exploration pipeline.

    Each iteration:
      1. Calls the LangGraph pipeline (specialists + fusion) on the current screenshot.
      2. If fusion returns GUESS → stop and return result.
      3. If fusion returns ROTATE/MOVE → execute action (fetch new Street View frame) → repeat.

    Stops early on GUESS or when max_iterations is exhausted.
    """
    game: Game = Game()
    game.reset(req.start_lat, req.start_lon, req.start_heading)
    async with AGENT_LOCK:
        return AGENT.run(game, req.max_iter)

asyncio.run(start_agent())