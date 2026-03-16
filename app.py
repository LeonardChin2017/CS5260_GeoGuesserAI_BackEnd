import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent import Agent, AnalysisResult
from game import Game

from pydantic import BaseModel

from util import log_event

AGENT_LOCK: asyncio.Lock = asyncio.Lock()
AGENT: Optional[Agent] = None


class AgentSession:
    """
    In-memory state for a single running agent/game session.

    The frontend owns Google Street View and sends screenshots +
    view metadata; the backend only tracks analysis/game state
    and issues high-level commands.
    """

    def __init__(self) -> None:
        self.running: bool = False
        self.iter: int = 0
        self.max_iter: int = 5
        self.game: Optional[Game] = None
        self.latest_analysis: Optional[AnalysisResult] = None
        self.last_action: Optional[str] = None
        self.last_frame: str = ""
        self.last_frame_at: Optional[str] = None
        self.captured_images: list[dict[str, Any]] = []
        self.last_observation: dict[str, Any] = {}
        self.command_seq: int = 0
        self.error: str = ""

    def reset(self, game: Game, max_iter: int = 5) -> None:
        self.running = True
        self.iter = 0
        self.max_iter = max_iter
        self.game = game
        self.latest_analysis = None
        self.last_action = None
        self.last_frame = ""
        self.last_frame_at = None
        self.captured_images = []
        self.last_observation = {}
        self.command_seq = 0
        self.error = ""


SESSION: AgentSession = AgentSession()

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
        if AGENT is None:
            AGENT = Agent()
        # Initialize a fresh game session; let Game choose a random Street View.
        game = Game()
        game.set_to_random_street_view()
        SESSION.reset(game, max_iter=5)
        log_event("[agent_start] New agent session created")
    return {
        "ok": True,
        "running": True,
        "steps": [],
        "game": {
            "view_lat": SESSION.game._cur_lat if SESSION.game else None,
            "view_lon": SESSION.game._cur_lon if SESSION.game else None,
            "heading": SESSION.game.heading if SESSION.game else None,
            "target_lat": SESSION.game._tar_lat if SESSION.game else None,
            "target_lon": SESSION.game._tar_lon if SESSION.game else None,
        },
    }


@app.post("/api/agent/stop")
async def stop_agent():
    global AGENT
    async with AGENT_LOCK:
        # Keep AGENT instance alive, but mark session as not running and clear
        # transient session state used by the frontend.
        SESSION.running = False
        SESSION.captured_images = []
        SESSION.last_observation = {}
        SESSION.latest_analysis = None
        SESSION.last_action = None
        SESSION.last_frame = ""
        SESSION.last_frame_at = None
        SESSION.error = ""
    return {"ok": True, "running": False, "steps": []}


@app.get("/api/agent/status")
async def agent_status():
    # Expose high-level session status.
    async with AGENT_LOCK:
        running = SESSION.running
        latest = SESSION.latest_analysis
        game = SESSION.game
        captured = SESSION.captured_images
        last_observation = SESSION.last_observation

    result_dict: Optional[dict[str, Any]] = None
    if latest is not None:
        result_dict = asdict(latest)

    game_snapshot: Optional[dict[str, Any]] = None
    if game is not None:
        # Expose only the fields used by the frontend.
        game_snapshot = {
            "view_lat": game._cur_lat,
            "view_lon": game._cur_lon,
            "heading": game.heading,
            # These may be populated once a guess is made; keep optional.
            "target_lat": game._tar_lat,
            "target_lon": game._tar_lon,
        }

    return {
        "running": running,
        "result": result_dict,
        "steps": [],
        "game": game_snapshot,
        "captured_images": captured,
        "last_observation": last_observation,
    }


@app.get("/api/agent/frame")
async def agent_frame():
    global AGENT
    out: dict[str, Any] = {}
    async with AGENT_LOCK:
        if SESSION.last_frame:
            out["frame"] = SESSION.last_frame
            out["frame_mime"] = "image/jpeg"
        if SESSION.game is not None:
            out["game"] = {
                "view_lat": SESSION.game._cur_lat,
                "view_lon": SESSION.game._cur_lon,
                "heading": SESSION.game.heading,
                "target_lat": SESSION.game._tar_lat,
                "target_lon": SESSION.game._tar_lon,
            }
        out["last_action"] = SESSION.last_action or ""
        out["last_frame_at"] = SESSION.last_frame_at
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


class ObservationRequest(BaseModel):
    command: str
    screenshot: Optional[str] = None
    screenshot_url: Optional[str] = None
    view_lat: Optional[float] = None
    view_lon: Optional[float] = None
    heading: Optional[float] = None
    pitch: Optional[float] = None
    guess_lat: Optional[float] = None
    guess_lon: Optional[float] = None
    command_seq: int


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
    # Legacy autopilot endpoint is not used by the frontend command/observation
    # loop anymore. Keep a minimal implementation for debugging.
    game: Game = Game()
    game.reset(req.start_lat, req.start_lon, req.start_heading)
    async with AGENT_LOCK:
        if AGENT is None:
            raise HTTPException(status_code=500, detail="Agent is not initialized")
        return AGENT.run(game, req.max_iter)


@app.get("/api/agent/next-command")
async def agent_next_command():
    """
    Decide the next high-level action for the frontend to take.
    """
    async with AGENT_LOCK:
        # If session is not running or game already has a final result, never issue
        # further commands.
        final_distance = None
        if SESSION.game is not None:
            final_distance = getattr(SESSION.game, "final_distance_km", None)
        if not SESSION.running or final_distance is not None:
            SESSION.running = False
            return {"running": False, "command": "idle", "command_seq": SESSION.command_seq}

        # First step in a session: always capture.
        if SESSION.latest_analysis is None:
            SESSION.command_seq += 1
            return {
                "running": True,
                "command": "capture",
                "command_seq": SESSION.command_seq,
            }

        action = SESSION.latest_analysis.action if SESSION.latest_analysis else {}
        action_type = str(action.get("type", "")).upper()
        degrees = float(action.get("degrees", 0.0) or 0.0)

        # If the agent has already decided to GUESS, treat that as terminal.
        if action_type == "GUESS":
            SESSION.running = False
            return {
                "running": False,
                "command": "idle",
                "command_seq": SESSION.command_seq,
            }

        command = "idle"
        if action_type == "ROTATE":
            command = "rotate_right" if degrees >= 0 else "rotate_left"
        elif action_type == "MOVE":
            # Map MOVE to a small right rotation as a simple surrogate.
            command = "rotate_right"
        else:
            command = "idle"

        if command == "idle":
            return {
                "running": SESSION.running,
                "command": "idle",
                "command_seq": SESSION.command_seq,
            }

        SESSION.command_seq += 1
        return {
            "running": SESSION.running,
            "command": command,
            "command_seq": SESSION.command_seq,
            "degrees": degrees,
        }


@app.post("/api/agent/observation")
async def agent_observation(req: ObservationRequest):
    """
    Process one observation from the frontend: a screenshot + view metadata.
    """
    async with AGENT_LOCK:
        if not SESSION.running or AGENT is None:
            raise HTTPException(status_code=400, detail="Agent session is not running")

        # Ignore out-of-order observations.
        if req.command_seq < SESSION.command_seq:
            return {
                "ok": False,
                "running": SESSION.running,
                "result": asdict(SESSION.latest_analysis) if SESSION.latest_analysis else None,
                "game": None,
                "steps": [],
                "captured_images": SESSION.captured_images,
                "last_observation": SESSION.last_observation,
            }

        # Normalize screenshot: accept full data URL or raw base64.
        screenshot_data = req.screenshot or ""
        if screenshot_data.startswith("data:"):
            try:
                screenshot_data = screenshot_data.split(",", 1)[1]
            except Exception:
                screenshot_data = ""

        error: Optional[str] = None
        capture_debug: Optional[str] = None

        if not screenshot_data:
            error = "Missing screenshot data"
            capture_debug = "observation_missing_screenshot"
        else:
            # Track raw frame for /api/agent/frame.
            SESSION.last_frame = screenshot_data
            SESSION.last_frame_at = datetime.now(timezone.utc).isoformat()
            # Store in captured_images gallery.
            SESSION.captured_images.append(
                {
                    "id": f"cap-{len(SESSION.captured_images) + 1}",
                    "captured_at": SESSION.last_frame_at,
                    "mime": "image/jpeg",
                    "data": screenshot_data,
                }
            )

        # Update game snapshot with latest view/guess metadata if available.
        if SESSION.game is None:
            SESSION.game = Game()
        if req.view_lat is not None and req.view_lon is not None:
            SESSION.game._cur_lat = req.view_lat
            SESSION.game._cur_lon = req.view_lon
        if req.heading is not None:
            SESSION.game.heading = req.heading

        # Run one analysis step when we have a screenshot.
        if not error and screenshot_data:
            result: AnalysisResult = AGENT.analyze(
                screenshot_data,
                float(req.heading or 0.0),
                SESSION.max_iter,
                SESSION.iter,
            )
            SESSION.latest_analysis = result
            SESSION.iter += 1
            SESSION.last_action = str(result.action.get("type", "")).upper()

            # If the agent already chose GUESS, treat this as a terminal step.
            if SESSION.last_action == "GUESS":
                try:
                    guess_lat = float(result.final_guess.get("lat"))
                    guess_lon = float(result.final_guess.get("lon"))
                    distance_km = SESSION.game.guess(guess_lat, guess_lon)
                    score = max(0.0, 5000.0 - distance_km * 10.0)
                    SESSION.running = False
                    setattr(SESSION.game, "final_distance_km", distance_km)
                    setattr(SESSION.game, "score", score)
                    setattr(SESSION.game, "guess_lat", guess_lat)
                    setattr(SESSION.game, "guess_lon", guess_lon)
                except Exception as exc:
                    error = f"Failed to compute guess score from agent final_guess: {exc}"
            elif SESSION.iter >= SESSION.max_iter:
                SESSION.running = False
        else:
            log_event(f"[agent_observation] Skipped analysis due to error: {error}")

        # If this observation carried a final guess, compute distance and stop.
        if req.command.lower() == "guess" and req.guess_lat is not None and req.guess_lon is not None:
            try:
                distance_km = SESSION.game.guess(req.guess_lat, req.guess_lon)
                # Attach simple scoring heuristic: higher score for smaller distance.
                score = max(0.0, 5000.0 - distance_km * 10.0)
                SESSION.running = False
                # Extend game snapshot with result fields expected by the frontend.
                setattr(SESSION.game, "final_distance_km", distance_km)
                setattr(SESSION.game, "score", score)
                setattr(SESSION.game, "guess_lat", req.guess_lat)
                setattr(SESSION.game, "guess_lon", req.guess_lon)
                # Mark latest_analysis as a GUESS-type action so status/next-command
                # logic consistently treats this as a terminal state.
                if SESSION.latest_analysis is not None:
                    SESSION.latest_analysis.action["type"] = "GUESS"
            except Exception as exc:
                error = f"Failed to compute guess score: {exc}"

        SESSION.last_observation = {
            "command": req.command,
            "error": error,
            "capture_debug": capture_debug,
        }

        game_snapshot: Optional[dict[str, Any]] = None
        if SESSION.game is not None:
            game_snapshot = {
                "view_lat": SESSION.game._cur_lat,
                "view_lon": SESSION.game._cur_lon,
                "heading": SESSION.game.heading,
                "target_lat": SESSION.game._tar_lat,
                "target_lon": SESSION.game._tar_lon,
                "final_distance_km": getattr(SESSION.game, "final_distance_km", None),
                "score": getattr(SESSION.game, "score", None),
                "guess_lat": getattr(SESSION.game, "guess_lat", None),
                "guess_lon": getattr(SESSION.game, "guess_lon", None),
            }

        return {
            "ok": error is None,
            "running": SESSION.running,
            "result": asdict(SESSION.latest_analysis) if SESSION.latest_analysis else None,
            "game": game_snapshot,
            "steps": [],
            "captured_images": SESSION.captured_images,
            "last_observation": SESSION.last_observation,
        }