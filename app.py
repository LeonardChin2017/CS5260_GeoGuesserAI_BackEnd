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
        # Do not start issuing capture/analyze commands before this UTC epoch seconds.
        self.analyze_not_before_s: float = 0.0
        # Final result state (kept in session; Game.guess() is stateless).
        self.final_distance_km: Optional[float] = None
        self.score: Optional[float] = None
        self.guess_lat: Optional[float] = None
        self.guess_lon: Optional[float] = None

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
        self.analyze_not_before_s = 0.0
        self.final_distance_km = None
        self.score = None
        self.guess_lat = None
        self.guess_lon = None


SESSION: AgentSession = AgentSession()


def _haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    r = 6371.0
    lat1 = math.radians(a_lat)
    lat2 = math.radians(b_lat)
    dlat = math.radians(b_lat - a_lat)
    dlon = math.radians(b_lon - a_lon)
    h = (math.sin(dlat / 2) ** 2) + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon / 2) ** 2)
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def _compute_distance_km(game: Game, guess_lat: float, guess_lon: float) -> float:
    """
    Prefer Game.guess() but guarantee a finite numeric distance.
    """
    d = float(game.guess(guess_lat, guess_lon))
    if math.isfinite(d):
        return d
    # Fallback: use haversine on stored target coordinates.
    return float(_haversine_km(guess_lat, guess_lon, float(game._tar_lat), float(game._tar_lon)))

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
        # Always show the initial image first; delay any analysis commands briefly.
        SESSION.analyze_not_before_s = datetime.now(timezone.utc).timestamp() + 2.0
        # Provide an initial frame immediately so the frontend can display the
        # view before any analysis/observations occur.
        try:
            SESSION.last_frame = game.render_image(size="640x640", timeout=20)
            SESSION.last_frame_at = datetime.now(timezone.utc).isoformat()
            SESSION.last_action = "INIT"
        except Exception as exc:
            log_event(f"[agent_start] initial frame fetch failed: {exc}")
            SESSION.last_frame = ""
            SESSION.last_frame_at = datetime.now(timezone.utc).isoformat()
            SESSION.last_action = "INIT_ERROR"
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
            # Final result fields (present once the agent commits a guess).
            "final_distance_km": SESSION.final_distance_km,
            "score": SESSION.score,
            "guess_lat": SESSION.guess_lat,
            "guess_lon": SESSION.guess_lon,
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
                "final_distance_km": SESSION.final_distance_km,
                "score": SESSION.score,
                "guess_lat": SESSION.guess_lat,
                "guess_lon": SESSION.guess_lon,
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



@app.post("/api/agent/stream-analyze")
async def agent_stream_analyze(req: AnalysisRequest):
    """
    Single-iteration streaming: runs the LangGraph pipeline on one screenshot.
    Yields Server-Sent Events (SSE) for each graph node update.
    """
    if AGENT is None:
        raise HTTPException(status_code=400, detail="Agent not started.")

    # Use session counters when a live run is active so next-command can advance.
    async with AGENT_LOCK:
        iter_value = SESSION.iter if SESSION.running else req.cur_iter
        max_iter_value = SESSION.max_iter if SESSION.running else req.max_iter

    async def graph_event_generator():
        final_belief_state: list[Any] = []
        final_action: dict[str, Any] = {}
        final_guess: dict[str, Any] = {}
        final_specialist_outputs: dict[str, Any] = {}
        final_error: str = ""

        for event in AGENT.stream_analyze(req.screenshot, req.heading, max_iter_value, iter_value):
            # Forward each SSE event to the client as-is.
            yield event

            # Also parse streamed updates so session state advances like /observation.
            if not event.startswith("data: "):
                continue
            raw_payload = event[6:].strip()
            try:
                payload = json.loads(raw_payload)
            except Exception:
                continue

            if isinstance(payload, dict) and isinstance(payload.get("error"), str):
                final_error = payload["error"]
                continue

            if not isinstance(payload, dict):
                continue

            for _node_name, state_update in payload.items():
                if not isinstance(state_update, dict):
                    continue
                if "specialist_outputs" in state_update and isinstance(state_update["specialist_outputs"], dict):
                    final_specialist_outputs.update(state_update["specialist_outputs"])
                if "belief_state" in state_update and isinstance(state_update["belief_state"], list):
                    final_belief_state = state_update["belief_state"]
                if "action" in state_update and isinstance(state_update["action"], dict):
                    final_action = state_update["action"]
                if "final_guess" in state_update and isinstance(state_update["final_guess"], dict):
                    final_guess = state_update["final_guess"]
                if "error" in state_update and isinstance(state_update["error"], str) and state_update["error"]:
                    final_error = state_update["error"]

        analysis_result = AnalysisResult(
            belief_state=final_belief_state,
            action=final_action,
            final_guess=final_guess,
            specialist_outputs=final_specialist_outputs,
            error=final_error,
        )

        async with AGENT_LOCK:
            # Keep session in sync with streamed analysis so next-command can terminate.
            SESSION.latest_analysis = analysis_result
            SESSION.iter += 1
            SESSION.last_action = str(analysis_result.action.get("type", "")).upper()
            if analysis_result.error:
                SESSION.error = analysis_result.error

            if SESSION.last_action == "GUESS":
                try:
                    guess_lat = float(analysis_result.final_guess.get("lat"))
                    guess_lon = float(analysis_result.final_guess.get("lon"))
                    if SESSION.game is not None:
                        distance_km = _compute_distance_km(SESSION.game, guess_lat, guess_lon)
                        score = max(0.0, 5000.0 - distance_km * 10.0)
                        SESSION.final_distance_km = float(distance_km)
                        SESSION.score = float(score)
                        SESSION.guess_lat = float(guess_lat)
                        SESSION.guess_lon = float(guess_lon)
                except Exception as exc:
                    SESSION.error = f"Failed to compute guess score from stream final_guess: {exc}"
                finally:
                    SESSION.running = False
            elif SESSION.iter >= SESSION.max_iter:
                SESSION.running = False

    return StreamingResponse(graph_event_generator(), media_type="text/event-stream")

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
        AGENT.game = game
        return AGENT.run(req.max_iter)


@app.get("/api/agent/next-command")
async def agent_next_command():
    """
    Decide the next high-level action for the frontend to take.
    """
    async with AGENT_LOCK:
        # If session is not running or game already has a final result, never issue
        # further commands.
        final_distance = None
        final_distance = SESSION.final_distance_km
        if not SESSION.running or final_distance is not None:
            SESSION.running = False
            return {"running": False, "command": "idle", "command_seq": SESSION.command_seq}

        # Ensure the frontend has time to show the initial frame before we start
        # issuing capture/analyze requests.
        now_s = datetime.now(timezone.utc).timestamp()
        if SESSION.analyze_not_before_s and now_s < SESSION.analyze_not_before_s:
            return {"running": True, "command": "idle", "command_seq": SESSION.command_seq}

        # First step in a session: always capture at the current backend view.
        if SESSION.latest_analysis is None:
            SESSION.command_seq += 1
            return {
                "running": True,
                "command": "capture",
                "command_seq": SESSION.command_seq,
                "view_lat": SESSION.game._cur_lat if SESSION.game else None,
                "view_lon": SESSION.game._cur_lon if SESSION.game else None,
                "heading": SESSION.game.heading if SESSION.game else None,
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

        # Backend is the source of truth for view state. Instead of telling the
        # frontend to "rotate", we update the backend view and request another
        # capture at the new view.
        if SESSION.game is None:
            SESSION.game = Game()

        if action_type == "ROTATE":
            try:
                SESSION.game.turn(delta_yaw=degrees, delta_pitch=0.0)
            except Exception as exc:
                log_event(f"[next_command] rotate failed: {exc}")
        elif action_type == "MOVE":
            try:
                SESSION.game.move_forward()
            except Exception as exc:
                log_event(f"[next_command] move failed: {exc}")
        elif action_type == "GUESS":
            SESSION.running = False
            return {"running": False, "command": "idle", "command_seq": SESSION.command_seq}

        # After applying the action (if any), request a new capture at the updated view.
        SESSION.command_seq += 1
        return {
            "running": SESSION.running,
            "command": "capture",
            "command_seq": SESSION.command_seq,
            "view_lat": SESSION.game._cur_lat,
            "view_lon": SESSION.game._cur_lon,
            "heading": SESSION.game.heading,
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
        # Backend is the source of truth for view state; the frontend may send
        # view metadata for debugging, but we do not overwrite backend state with it.

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
                    distance_km = _compute_distance_km(SESSION.game, guess_lat, guess_lon)
                    score = max(0.0, 5000.0 - distance_km * 10.0)
                    SESSION.running = False
                    SESSION.final_distance_km = float(distance_km)
                    SESSION.score = float(score)
                    SESSION.guess_lat = float(guess_lat)
                    SESSION.guess_lon = float(guess_lon)
                except Exception as exc:
                    error = f"Failed to compute guess score from agent final_guess: {exc}"
            elif SESSION.iter >= SESSION.max_iter:
                SESSION.running = False
        else:
            log_event(f"[agent_observation] Skipped analysis due to error: {error}")

        # If this observation carried a final guess, compute distance and stop.
        if req.command.lower() == "guess" and req.guess_lat is not None and req.guess_lon is not None:
            try:
                distance_km = _compute_distance_km(SESSION.game, float(req.guess_lat), float(req.guess_lon))
                # Attach simple scoring heuristic: higher score for smaller distance.
                score = max(0.0, 5000.0 - distance_km * 10.0)
                SESSION.running = False
                SESSION.final_distance_km = float(distance_km)
                SESSION.score = float(score)
                SESSION.guess_lat = float(req.guess_lat)
                SESSION.guess_lon = float(req.guess_lon)
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
                "final_distance_km": SESSION.final_distance_km,
                "score": SESSION.score,
                "guess_lat": SESSION.guess_lat,
                "guess_lon": SESSION.guess_lon,
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