import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from agent import Agent, AnalysisResult
from game import Game

from pydantic import BaseModel

from util import log_event

AGENT_LOCK: asyncio.Lock = asyncio.Lock()
AGENT: Optional[Agent] = None
app = FastAPI()

# Allow the Vite dev server (frontend) to call this API from http://localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        game = Game()
        if AGENT is None:
            AGENT = Agent(game=game)
        else:
            AGENT.game = game
        game.set_to_random_street_view()
    return {"ok": True}


@app.post("/api/agent/stop")
async def stop_agent():
    global AGENT
    async with AGENT_LOCK:
        AGENT = None
    return {"ok": True}


@app.get("/api/agent/status")
async def agent_status():
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
    global AGENT
    async with AGENT_LOCK:
        if AGENT is None:
            raise HTTPException(status_code=400, detail="Agent not started.")
        result: AnalysisResult = AGENT.analyze(req.screenshot, req.heading, req.max_iter, req.cur_iter)
    return asdict(result)



@app.post("/api/agent/stream-analyze")
async def agent_stream_analyze(req: AnalysisRequest):
    """
    Single-iteration streaming: runs the LangGraph pipeline on one screenshot.
    Yields Server-Sent Events (SSE) for each graph node update.
    """
    # Read the current agent under lock to avoid races.
    global AGENT
    async with AGENT_LOCK:
        if AGENT is None:
            raise HTTPException(status_code=400, detail="Agent not started.")

    async def graph_event_generator():
        for event in AGENT.stream_analyze(req.screenshot, req.heading, req.max_iter, req.cur_iter):
            yield event
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(graph_event_generator(), media_type="text/event-stream")

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
    global AGENT
    async with AGENT_LOCK:
        if AGENT is None:
            raise HTTPException(status_code=400, detail="Agent not started.")
        AGENT.game.reset(req.start_lat, req.start_lon, req.start_heading)
        return AGENT.run(req.max_iter)


@app.post("/api/agent/stream-run")
async def agent_stream_run(req: RunRequest):
    """
    Full autopilot loop with streaming updates: runs the multi-iteration exploration pipeline.

    Each iteration:
      1. Calls the LangGraph pipeline (specialists + fusion) on the current screenshot.
      2. Streams SSE updates for each graph node execution.
      3. If fusion returns GUESS → stop and return result.
      4. If fusion returns ROTATE/MOVE → execute action (fetch new Street View frame) → repeat.

    Stops early on GUESS or when max_iterations is exhausted.
    """
    global AGENT
    async with AGENT_LOCK:
        if AGENT is None:
            raise HTTPException(status_code=500, detail="Agent is not initialized")
        AGENT.game.reset(req.start_lat, req.start_lon, req.start_heading)

    async def graph_event_generator():
        for event in AGENT.stream_run(req.max_iter):
            yield event
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(graph_event_generator(), media_type="text/event-stream")
