import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
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
    return {"ok": True}


@app.post("/api/agent/stop")
async def stop_agent():
    global AGENT
    async with AGENT_LOCK:
        if AGENT is not None:
            AGENT.request_stream_run_cancel()
            AGENT.request_stream_analyze_cancel()
    return {"ok": True}

@app.post("/api/agent/new-streetview")
async def agent_new_streetview():
    global AGENT
    game_state: dict[str, Any] = {}
    async with AGENT_LOCK:
        if AGENT is None:
            raise HTTPException(status_code=400, detail="Agent not started.")
        if AGENT.game is None:
            raise HTTPException(status_code=500, detail="Agent game is not initialized.")
        AGENT.reset_runtime_state()
        AGENT.game.set_to_random_street_view()
        AGENT.render_image(None)
        game_state = AGENT.get_ui_game_state()
    return {"ok": True, "game": game_state}

@app.get("/api/agent/status")
async def agent_status():
    game_state: dict[str, Any] = {}
    last_action: str | None = None
    last_frame_at: str | None = None
    async with AGENT_LOCK:
        running = AGENT is not None
        if AGENT is not None:
            game_state = AGENT.get_ui_game_state()
            last_action = AGENT.last_action
            last_frame_at = AGENT.last_frame_at
    out: dict[str, Any] = {"running": running, "game": game_state}
    if last_action:
        out["last_action"] = last_action
    if last_frame_at:
        out["last_frame_at"] = last_frame_at
    return out


@app.get("/api/agent/frame")
async def agent_frame():
    global AGENT
    out: dict[str, Any] = {}
    async with AGENT_LOCK:
        if AGENT is None:
            return out
        if len(AGENT.frame) > 0:
            out["frame"] = AGENT.frame
            out["frame_mime"] = "image/jpeg"
        out["game"] = AGENT.get_ui_game_state()
        if AGENT.last_action:
            out["last_action"] = AGENT.last_action
        if AGENT.last_frame_at:
            out["last_frame_at"] = AGENT.last_frame_at
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
        AGENT.reset_runtime_state()
        result: AnalysisResult = AGENT.analyze(req.screenshot, req.heading, req.max_iter, req.cur_iter)
    return asdict(result)



@app.post("/api/agent/stream-analyze")
async def agent_stream_analyze(req: AnalysisRequest, request: Request):
    """
    Single-iteration streaming: runs the LangGraph pipeline on one screenshot.
    Yields Server-Sent Events (SSE) for each graph node update.
    """
    # Read the current agent under lock to avoid races.
    global AGENT
    async with AGENT_LOCK:
        if AGENT is None:
            raise HTTPException(status_code=400, detail="Agent not started.")
        AGENT.reset_runtime_state()

    async def graph_event_generator():
        for event in AGENT.stream_analyze(req.screenshot, req.heading, req.max_iter, req.cur_iter):
            if await request.is_disconnected():
                AGENT.request_stream_run_cancel()
                break
            yield event
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(graph_event_generator(), media_type="text/event-stream")

class RunRequest(BaseModel):
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
        AGENT.reset_runtime_state()
        return AGENT.run(req.max_iter)


@app.post("/api/agent/stream-run")
async def agent_stream_run(req: RunRequest, request: Request):
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
        AGENT.reset_runtime_state()

    async def graph_event_generator():
        for event in AGENT.stream_run(req.max_iter):
            if await request.is_disconnected():
                AGENT.request_stream_run_cancel()
                break
            yield event
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(graph_event_generator(), media_type="text/event-stream")
