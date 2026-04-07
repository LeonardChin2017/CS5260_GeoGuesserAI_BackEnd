import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from agent import Agent, AnalysisResult
from game import Game

from pydantic import BaseModel, Field

from util import log_event

AGENT_LOCK: asyncio.Lock = asyncio.Lock()
AGENT: Optional[Agent] = None
app = FastAPI()

def _get_cors_origins() -> list[str]:
    """
    Return allowed CORS origins from env (comma-separated).

    Example:
      CORS_ALLOW_ORIGINS=https://your-app.vercel.app,https://app.example.com
    """
    origins = {"http://localhost:5173", "http://127.0.0.1:5173"}
    raw = (os.getenv("CORS_ALLOW_ORIGINS") or "").strip()
    if raw:
        origins.update(origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip())
    return sorted(origins)


CORS_ORIGINS = _get_cors_origins()

# Default: local Vite dev server. In production, set CORS_ALLOW_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    # Wildcard origins and credentials are incompatible in browsers.
    allow_credentials="*" not in CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_agent_ready_for_run() -> None:
    """
    Make sure AGENT exists and has an initialized Street View state.

    This keeps /api/agent/stream-run resilient to out-of-order client calls
    (for example, stop/start/new-streetview racing each other over network).
    """
    global AGENT

    if AGENT is None:
        AGENT = Agent(game=Game())
    elif AGENT.game is None:
        AGENT.game = Game()
        AGENT.reset_runtime_state()

    game_state = AGENT.game.get_state()
    needs_bootstrap = any(
        isinstance(game_state.get(key), float) and math.isnan(game_state.get(key))
        for key in ("view_lat", "view_lon", "heading")
    )
    if needs_bootstrap:
        AGENT.game.set_to_random_street_view()
        AGENT.render_image(None)


def _agent_snapshot_payload(agent: Agent, include_frame: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {"game": agent.get_ui_game_state()}
    if include_frame and len(agent.frame) > 0:
        payload["frame"] = agent.frame
        payload["frame_mime"] = "image/jpeg"
    if agent.last_action:
        payload["last_action"] = agent.last_action
    if agent.last_frame_at:
        payload["last_frame_at"] = agent.last_frame_at
    return payload


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
            AGENT.reset_runtime_state()
    return {"ok": True}


@app.post("/api/agent/stop")
async def stop_agent():
    global AGENT
    async with AGENT_LOCK:
        AGENT = None
    return {"ok": True}

@app.post("/api/agent/new-streetview")
async def agent_new_streetview():
    global AGENT
    async with AGENT_LOCK:
        if AGENT is None:
            raise HTTPException(status_code=400, detail="Agent not started.")
        if AGENT.game is None:
            raise HTTPException(status_code=500, detail="Agent game is not initialized.")
        AGENT.reset_runtime_state()
        AGENT.game.set_to_random_street_view()
        AGENT.render_image(None)
        snapshot = _agent_snapshot_payload(AGENT)
    return {"ok": True, **snapshot}


class TurnRequest(BaseModel):
    degrees: float = Field(default=90.0)


class MoveForwardRequest(BaseModel):
    distance_m: float = Field(default=20.0, gt=0.0)


@app.post("/api/agent/turn")
async def agent_turn(req: TurnRequest):
    global AGENT
    async with AGENT_LOCK:
        if AGENT is None:
            raise HTTPException(status_code=400, detail="Agent not started.")
        if AGENT.game is None:
            raise HTTPException(status_code=500, detail="Agent game is not initialized.")
        try:
            AGENT.game.turn(delta_yaw=req.degrees)
            AGENT.last_action = None
            AGENT.render_image(None)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to turn street view: {exc}")
        snapshot = _agent_snapshot_payload(AGENT)
    return {"ok": True, **snapshot}


@app.post("/api/agent/move-forward")
async def agent_move_forward(req: MoveForwardRequest):
    global AGENT
    async with AGENT_LOCK:
        if AGENT is None:
            raise HTTPException(status_code=400, detail="Agent not started.")
        if AGENT.game is None:
            raise HTTPException(status_code=500, detail="Agent game is not initialized.")
        try:
            AGENT.game.move_forward(distance_m=req.distance_m)
            AGENT.last_action = None
            AGENT.render_image(None)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to move street view forward: {exc}")
        snapshot = _agent_snapshot_payload(AGENT)
    return {"ok": True, **snapshot}


@app.post("/api/agent/zoom-in")
async def agent_zoom_in():
    global AGENT
    async with AGENT_LOCK:
        if AGENT is None:
            raise HTTPException(status_code=400, detail="Agent not started.")
        if AGENT.game is None:
            raise HTTPException(status_code=500, detail="Agent game is not initialized.")
        try:
            AGENT.game.zoom_in(step=1)
            AGENT.last_action = None
            AGENT.render_image(None)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to zoom in street view: {exc}")
        snapshot = _agent_snapshot_payload(AGENT)
    return {"ok": True, **snapshot}


@app.post("/api/agent/zoom-out")
async def agent_zoom_out():
    global AGENT
    async with AGENT_LOCK:
        if AGENT is None:
            raise HTTPException(status_code=400, detail="Agent not started.")
        if AGENT.game is None:
            raise HTTPException(status_code=500, detail="Agent game is not initialized.")
        try:
            AGENT.game.zoom_out(step=1)
            AGENT.last_action = None
            AGENT.render_image(None)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to zoom out street view: {exc}")
        snapshot = _agent_snapshot_payload(AGENT)
    return {"ok": True, **snapshot}

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
        out = _agent_snapshot_payload(AGENT)
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
        try:
            _ensure_agent_ready_for_run()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Agent bootstrap failed: {exc}")
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
        try:
            _ensure_agent_ready_for_run()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Agent bootstrap failed: {exc}")

    async def graph_event_generator():
        for event in AGENT.stream_run(req.max_iter):
            yield event
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(graph_event_generator(), media_type="text/event-stream")
