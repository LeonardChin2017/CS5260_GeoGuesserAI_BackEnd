import asyncio
import base64
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from math import nan, isnan
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graphs.agent_runner import run_langgraph_agent, _PIPELINE_STEPS
from util import _0_if_nan, log_debug

load_dotenv()

from graphs.geoguessr_graph import geo_graph
from graphs.state import GeoState
from graphs.action_executor import GameView, execute_action

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR = BASE_DIR / "prompts"
CAPTURED_IMAGES_DIR = DATA_DIR / "captures"
CAPTURED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()


CAPTURE_PIPELINE_VERSION = "capture-v3-ext-sniff"


AGENT_LOCK = asyncio.Lock()


@dataclass
class GameState:
    heading: float = nan
    view_lon: float = nan
    view_lat: float = nan
    guess_lon: float = nan
    guess_lat: float = nan
    confidence: float = nan
    detected_clues: list[str] = field(default_factory=list)
    clues: list[str] = field(default_factory=list)
    best_country_guess: str = ''
    target_country: str = ''
    target_lat: float = nan
    target_lon: float = nan
    final_distance_km: float = nan
    score: int = -1


@dataclass
class AgentState:
    running: bool = False
    stop: bool = False
    steps: list[dict[str, Any]] = field(default_factory=list)
    frame: Optional[str] = None
    frame_mime: str = "image/svg+xml"
    error: Optional[str] = None
    last_action: Optional[str] = None
    last_frame_at: Optional[str] = None
    game: Optional[GameState] = None
    pending_command: Optional[str] = None
    command_seq: int = 0
    last_observation: Optional[dict[str, Any]] = None
    captured_images: list[dict[str, Any]] = field(default_factory=list)


AGENT_STATE: AgentState = AgentState()
AGENT_QUEUE: asyncio.Queue[str] = asyncio.Queue()
AGENT_TASK_STARTED: bool = False
AGENT_STEP_SEQUENCE: list[tuple[str, str]] = [
    ("capture", "Capture current clues"),
    ("rotate_left", "Rotate view left"),
    ("rotate_right", "Rotate view right"),
    ("pan_random", "Pan around for landmarks"),
    ("move_forward", "Move forward to gather more clues"),
    ("detect", "Detect signs, language, and road markings"),
    ("match", "Match clues to likely country"),
    ("guess", "Place guess and submit"),
]

AGENT_COMMAND_FLOW: list[str] = ["capture", "rotate_left", "rotate_right", "guess"]

GEO_LOCATIONS = [
    {
        "name": "Shibuya Crossing",
        "country": "Japan",
        "lat": 35.6595,
        "lon": 139.7005,
        "clues": ["Neon signs", "Left-hand traffic", "Dense urban crossing", "Japanese writing"],
    },
    {
        "name": "Copacabana",
        "country": "Brazil",
        "lat": -22.9711,
        "lon": -43.1822,
        "clues": ["Beach promenade", "Portuguese text", "Palm trees", "Atlantic coastline"],
    },
    {
        "name": "Table Mountain",
        "country": "South Africa",
        "lat": -33.9628,
        "lon": 18.4098,
        "clues": ["Flat-topped mountain", "English road signs", "Coastal city", "Southern hemisphere sun"],
    },
    {
        "name": "Reykjavik Harbor",
        "country": "Iceland",
        "lat": 64.1466,
        "lon": -21.9426,
        "clues": ["Nordic architecture", "Cold weather", "Sparse trees", "Volcanic landscape"],
    },
]


async def _agent_snapshot() -> dict[str, Any]:
    log_debug("agent_snapshot called")
    async with AGENT_LOCK:
        steps: list[dict[str, Any]] = [dict(step) for step in AGENT_STATE.steps]
        game: Optional[dict[str, Any]] = asdict(AGENT_STATE.game) if isinstance(AGENT_STATE.game, GameState) else None
        captured_images: list[dict[str, Any]] = [dict(img) for img in (AGENT_STATE.captured_images or [])]
        return {
            "running": bool(AGENT_STATE.running),
            "steps": steps,
            "error": AGENT_STATE.error,
            "last_action": AGENT_STATE.last_action,
            "last_frame_at": AGENT_STATE.last_frame_at,
            "game": game,
            "pending_command": AGENT_STATE.pending_command,
            "command_seq": int(AGENT_STATE.command_seq or 0),
            "last_observation": AGENT_STATE.last_observation,
            "captured_images": captured_images,
        }


async def _agent_set_steps(steps: list[dict[str, Any]]) -> None:
    async with AGENT_LOCK:
        AGENT_STATE.steps = steps


async def _agent_set_frame(frame_b64: Optional[str], frame_mime: str = "image/svg+xml") -> None:
    async with AGENT_LOCK:
        AGENT_STATE.frame = frame_b64
        AGENT_STATE.frame_mime = frame_mime
        AGENT_STATE.last_frame_at = datetime.now(UTC).isoformat()


async def _agent_set_error(message: Optional[str]) -> None:
    async with AGENT_LOCK:
        AGENT_STATE.error = message


async def _agent_set_action(action: Optional[str]) -> None:
    async with AGENT_LOCK:
        AGENT_STATE.last_action = action


def _parse_data_url_image(data_url: str) -> Optional[dict[str, str]]:
    raw = (data_url or "").strip()
    if not raw.startswith("data:image/"):
        return None
    comma = raw.find(",")
    if comma <= 0:
        return None
    meta = raw[5:comma]
    payload = raw[comma + 1:]
    if ";base64" not in meta or not payload:
        return None
    mime = meta.split(";")[0].strip().lower()
    if not mime.startswith("image/"):
        return None
    try:
        base64.b64decode(payload, validate=True)
    except Exception:
        return None
    return {"mime": mime, "data": payload}


def _mime_to_extension(mime: str) -> str:
    m = (mime or "").split(";", 1)[0].strip().lower()
    if m in ("image/jpeg", "image/jpg", "image/pjpeg"):
        return ".jpg"
    if m == "image/png":
        return ".png"
    if m == "image/webp":
        return ".webp"
    if m == "image/gif":
        return ".gif"
    if m in ("image/svg+xml", "image/svg"):
        return ".svg"
    if m == "image/bmp":
        return ".bmp"
    if m == "image/avif":
        return ".avif"
    return ".bin"


def _guess_extension_from_bytes(raw: bytes) -> Optional[str]:
    if not raw:
        return None
    if raw.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if raw.startswith(b"BM"):
        return ".bmp"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return ".webp"
    if len(raw) > 12 and raw[4:12] == b"ftypavif":
        return ".avif"
    head = raw[:512].lstrip()
    lower_head = head.lower()
    if lower_head.startswith(b"<svg") or (lower_head.startswith(b"<?xml") and b"<svg" in lower_head):
        return ".svg"
    return None


def _persist_capture_image(image_id: str, mime: str, data_b64: str) -> Optional[str]:
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except Exception:
        return None
    if not raw:
        return None
    ext = _mime_to_extension(mime)
    if ext == ".bin":
        ext = _guess_extension_from_bytes(raw) or ".bin"
    file_name = f"{image_id}{ext}"
    file_path = CAPTURED_IMAGES_DIR / file_name
    try:
        file_path.write_bytes(raw)
    except Exception:
        return None
    return f"/captured_images/{file_name}"


def _step_index(step_id: str) -> int:
    for idx, (sid, _) in enumerate(AGENT_STEP_SEQUENCE):
        if sid == step_id:
            return idx
    return -1


async def _mark_step_running(step_id: str) -> None:
    idx = _step_index(step_id)
    if idx < 0:
        return
    async with AGENT_LOCK:
        if not AGENT_STATE.steps:
            AGENT_STATE.steps = [{"id": sid, "message": msg, "status": "pending"} for sid, msg in AGENT_STEP_SEQUENCE]
        AGENT_STATE.steps[idx]["status"] = "running"


async def _mark_step_done(step_id: str) -> None:
    idx = _step_index(step_id)
    if idx < 0:
        return
    async with AGENT_LOCK:
        if not AGENT_STATE.steps:
            AGENT_STATE.steps = [{"id": sid, "message": msg, "status": "pending"} for sid, msg in AGENT_STEP_SEQUENCE]
        AGENT_STATE.steps[idx]["status"] = "done"


async def _set_pending_command(command: Optional[str]) -> None:
    async with AGENT_LOCK:
        AGENT_STATE.pending_command = command
        AGENT_STATE.command_seq = int(AGENT_STATE.command_seq or 0) + 1


def _build_game_state() -> GameState:
    target = random.choice(GEO_LOCATIONS)
    return GameState(
        target_country=target["country"],
        target_lat=target["lat"],
        target_lon=target["lon"],
        clues=list(target["clues"]),
        guess_lat=random.uniform(-60.0, 65.0),
        guess_lon=random.uniform(-170.0, 170.0),
        view_lat=target["lat"],
        view_lon=target["lon"],
        heading=random.uniform(0.0, 359.0),
        confidence=0.2
    )


async def _mark_step_progress(step_index: int) -> None:
    async with AGENT_LOCK:
        if not AGENT_STATE.steps:
            return
        if 0 <= step_index < len(AGENT_STATE.steps):
            AGENT_STATE.steps[step_index]["status"] = "done"
            if step_index + 1 < len(AGENT_STATE.steps):
                AGENT_STATE.steps[step_index + 1]["status"] = "running"


async def _apply_agent_action(step_id: str) -> None:
    async with AGENT_LOCK:
        game: Optional[GameState] = AGENT_STATE.game
    if game is None:
        return
    if step_id == "rotate_left":
        game.heading = _wrap_heading(game.heading - random.uniform(25.0, 70.0))
    elif step_id == "rotate_right":
        game.heading = _wrap_heading(game.heading + random.uniform(25.0, 70.0))
    elif step_id == "pan_random":
        game.view_lon = max(-180.0, min(180.0, game.view_lon + random.uniform(-0.002, 0.002)))
        game.view_lat = max(-85.0, min(85.0, game.view_lat + random.uniform(-0.001, 0.001)))
        game.guess_lon = max(-180.0, min(180.0, game.guess_lon + random.uniform(-16.0, 16.0)))
        game.guess_lat = max(-85.0, min(85.0, game.guess_lat + random.uniform(-8.0, 8.0)))
    elif step_id == "move_forward":
        heading: float = game.heading
        lat_step: float = 0.8 * (1 if 0 <= heading < 180 else -1)
        lon_step: float = 1.2 * (1 if heading <= 90 or heading >= 270 else -1)
        game.view_lat = max(-85.0, min(85.0, game.view_lat + (lat_step * 0.001)))
        game.view_lon = max(-180.0, min(180.0, game.view_lon + (lon_step * 0.001)))
        game.guess_lat = max(-85.0, min(85.0, game.guess_lat + lat_step))
        game.guess_lon = max(-180.0, min(180.0, game.guess_lon + lon_step))
        game.confidence = min(0.95, game.confidence + 0.08)
    elif step_id == "detect":
        clues: list[str] = list(game.clues)
        seen: list[str] = list(game.detected_clues)
        if len(seen) < len(clues):
            seen.append(clues[len(seen)])  # TODO verify not a bug
        game.detected_clues = seen
        game.confidence = min(0.98, game.confidence + 0.18)
    elif step_id == "match":
        game.best_country_guess = game.target_country
        game.confidence = min(0.99, game.confidence + 0.15)
    elif step_id == "guess":
        target_lat: float = game.target_lat
        target_lon: float = game.target_lon
        if isnan(target_lat) or isnan(target_lon):
            game.final_distance_km = nan
            game.score = -1
        else:
            distance = _distance_km(
                _0_if_nan(game.guess_lat),
                _0_if_nan(game.guess_lon),
                target_lat,
                target_lon,
            )
            game.final_distance_km = distance
            game.score = _score_from_distance(distance)
    async with AGENT_LOCK:
        AGENT_STATE.game = game


async def _agent_worker() -> None:
    min_delay = float(os.getenv("AGENT_STEP_DELAY_MIN_SECONDS", "0.8"))
    max_delay = float(os.getenv("AGENT_STEP_DELAY_MAX_SECONDS", "1.8"))
    steps_template = [{"id": sid, "message": msg, "status": "pending"} for sid, msg in AGENT_STEP_SEQUENCE]
    current_step_index = -1
    step_started_at: Optional[float] = None
    current_step_delay: Optional[float] = None

    async def set_steps_for_start() -> None:
        nonlocal current_step_index, step_started_at, current_step_delay
        steps = [dict(s) for s in steps_template]
        if steps:
            steps[0]["status"] = "running"
            current_step_index = 0
            step_started_at = time.monotonic()
            current_step_delay = random.uniform(min_delay, max_delay)
        await _agent_set_steps(steps)

    while True:
        try:
            cmd = AGENT_QUEUE.get_nowait()
        except asyncio.QueueEmpty:
            cmd = None

        if cmd == "start":
            await _agent_set_error(None)
            async with AGENT_LOCK:
                AGENT_STATE.game = _build_game_state()
            async with AGENT_LOCK:
                AGENT_STATE.running = True
                AGENT_STATE.stop = False
            await set_steps_for_start()
        elif cmd == "stop":
            async with AGENT_LOCK:
                AGENT_STATE.running = False
                AGENT_STATE.stop = True
            current_step_index = -1
            step_started_at = None

        async with AGENT_LOCK:
            running = AGENT_STATE.running
            should_stop = AGENT_STATE.stop

        if running and not should_stop and current_step_index >= 0:
            if step_started_at is None or current_step_delay is None:
                step_started_at = time.monotonic()
                current_step_delay = random.uniform(min_delay, max_delay)
            elapsed = time.monotonic() - step_started_at
            if elapsed >= current_step_delay and current_step_index < len(AGENT_STEP_SEQUENCE):
                sid, _ = AGENT_STEP_SEQUENCE[current_step_index]
                try:
                    await _agent_set_action(sid)
                    await _apply_agent_action(sid)
                except Exception as exc:
                    await _agent_set_error(f"Action failed: {exc}")
                await _mark_step_progress(current_step_index)
                current_step_index += 1
                step_started_at = time.monotonic()
                current_step_delay = random.uniform(min_delay, max_delay)
                if current_step_index >= len(AGENT_STEP_SEQUENCE):
                    async with AGENT_LOCK:
                        AGENT_STATE.running = False
                    current_step_index = -1
                    step_started_at = None

        await asyncio.sleep(0.2)


def _start_agent_thread() -> None:
    global AGENT_TASK_STARTED
    if AGENT_TASK_STARTED:
        return
    asyncio.create_task(_agent_worker())
    AGENT_TASK_STARTED = True


@dataclass
class GameActionRequest(BaseModel):
    action: str = ''
    guess_lat: Optional[float] = None
    guess_lon: Optional[float] = None


@dataclass
class AgentObservationRequest(BaseModel):
    command: str = ''
    screenshot: Optional[str] = None
    screenshot_url: Optional[str] = None
    maps_api_key: Optional[str] = None
    view_lat: Optional[float] = None
    view_lon: Optional[float] = None
    heading: Optional[float] = None
    guess_lat: Optional[float] = None
    guess_lon: Optional[float] = None
    error: Optional[str] = None


app = FastAPI()


@app.get("/")
def root():
    return {"service": "GGSolver-backend", "message": "API only. Try GET /health."}


@app.get("/health")
def health():
    return {"ok": True}


class AnalyzeRequest(BaseModel):
    screenshot: str  # base64-encoded image (data URL or raw base64)
    max_iterations: int = 5


@app.post("/api/agent/analyze")
async def agent_analyze(req: AnalyzeRequest):
    """
    Single-iteration: run the LangGraph pipeline on one screenshot.
    Returns belief_state, action, action_history, final_guess.
    Useful for debugging individual iterations.
    """
    initial_state: GeoState = {
        "screenshot": req.screenshot,
        "iteration": 0,
        "max_iterations": req.max_iterations,
        "specialist_outputs": {},
        "belief_state": [],
        "action": {},
        "action_history": [],
        "final_guess": None,
        "error": None,
    }
    try:
        result = geo_graph.invoke(initial_state)
        return {
            "belief_state": result.get("belief_state", []),
            "action": result.get("action", {}),
            "action_history": result.get("action_history", []),
            "final_guess": result.get("final_guess"),
            "specialist_outputs": result.get("specialist_outputs", {}),
            "iteration": result.get("iteration", 0),
            "error": result.get("error"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RunRequest(BaseModel):
    screenshot: str  # initial base64 screenshot
    start_lat: float  # starting Street View position
    start_lon: float
    start_heading: int = 0
    max_iterations: int = 5


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
    maps_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    view = GameView(req.start_lat, req.start_lon, req.start_heading)
    screenshot = req.screenshot

    belief_state: list = []
    action_history: list = []
    result: dict = {}
    errors: list = []

    for iteration in range(req.max_iterations):
        state: GeoState = {
            "screenshot": screenshot,
            "iteration": iteration,
            "max_iterations": req.max_iterations,
            "specialist_outputs": {},  # fresh each iteration
            "belief_state": belief_state,  # carry forward
            "action": {},
            "action_history": action_history,
            "final_guess": None,
            "error": None,
        }

        try:
            result = geo_graph.invoke(state)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Graph error at iteration {iteration}: {e}")

        belief_state = result.get("belief_state", belief_state)
        action_history = result.get("action_history", action_history)
        action = result.get("action", {})

        if result.get("error"):
            errors.append(result["error"])

        # Committed a guess — done
        if action.get("type") == "GUESS":
            break

        # Explore — execute action and fetch new screenshot
        if not maps_key:
            errors.append("GOOGLE_MAPS_API_KEY not set; cannot fetch new Street View frame")
            break

        try:
            screenshot, view = execute_action(action, view, maps_key)
        except Exception as e:
            errors.append(f"Action execution failed at iteration {iteration}: {e}")
            break

    return {
        "final_guess": result.get("final_guess"),
        "belief_state": result.get("belief_state", []),
        "action_history": action_history,
        "iterations_used": result.get("iteration", 0),
        "final_view": view.to_dict(),
        "errors": errors if errors else None,
    }


@app.get("/api/agent/captures")
def list_captures():
    files = []
    try:
        for p in sorted(CAPTURED_IMAGES_DIR.glob("*")):
            if p.is_file():
                files.append(
                    {
                        "name": p.name,
                        "size_bytes": p.stat().st_size,
                        "url": f"/captured_images/{p.name}",
                    }
                )
    except Exception:
        files = []
    return {
        "version": CAPTURE_PIPELINE_VERSION,
        "base_dir": str(BASE_DIR),
        "data_dir": str(DATA_DIR),
        "dir": str(CAPTURED_IMAGES_DIR),
        "count": len(files),
        "files": files,
    }


@app.on_event("startup")
async def on_startup() -> None:
    _start_agent_thread()
    async with AGENT_LOCK:
        if not AGENT_STATE.game is not None:
            AGENT_STATE.game = _build_game_state()


@app.post("/api/agent/start")
async def start_agent():
    async with AGENT_LOCK:
        if AGENT_STATE.running:
            return await _agent_snapshot()
        game: GameState = _build_game_state()
        AGENT_STATE.running = True
        AGENT_STATE.stop = False
        AGENT_STATE.error = None
        AGENT_STATE.game = game
        AGENT_STATE.steps = [{"id": sid, "message": msg, "status": "pending"} for sid, msg in _PIPELINE_STEPS]
        AGENT_STATE.last_observation = None
        AGENT_STATE.captured_images = []

    # Launch background LangGraph runner
    asyncio.create_task(run_langgraph_agent(
        agent_state=AGENT_STATE,
        agent_lock=AGENT_LOCK,
        start_lat=game.view_lat,
        start_lon=game.view_lon,
        start_heading=game.heading,
        max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "5")),
    ))

    return await _agent_snapshot()


@app.post("/api/agent/stop")
async def stop_agent():
    async with AGENT_LOCK:
        AGENT_STATE.stop = True
        AGENT_STATE.running = False
        AGENT_STATE.pending_command = None
        AGENT_STATE.steps = []
        AGENT_STATE.last_observation = None
        AGENT_STATE.last_action = None
        AGENT_STATE.captured_images = []
    return await _agent_snapshot()


@app.get("/api/agent/next-command")
async def agent_next_command():
    async with AGENT_LOCK:
        return {
            "running": bool(AGENT_STATE.running),
            "command": AGENT_STATE.pending_command,
            "command_seq": int(AGENT_STATE.command_seq or 0),
        }


@app.post("/api/agent/observation")
async def agent_observation(body: AgentObservationRequest):
    command = (body.command or "").strip().lower()
    if not command:
        raise HTTPException(status_code=400, detail="command required")

    async with AGENT_LOCK:
        running = bool(AGENT_STATE.running)
        pending = AGENT_STATE.pending_command
    if not running:
        raise HTTPException(status_code=409, detail="agent is not running")
    if pending != command:
        raise HTTPException(status_code=409, detail=f"unexpected command '{command}', pending '{pending}'")

    capture_debug: Optional[str] = None
    async with AGENT_LOCK:
        game: GameState = GameState() if AGENT_STATE.game is None else AGENT_STATE.game
        if body.view_lat is not None:
            game.view_lat = float(max(-85.0, min(85.0, body.view_lat)))
        if body.view_lon is not None:
            game.view_lon = float(max(-180.0, min(180.0, body.view_lon)))
        if body.heading is not None:
            game.heading = _wrap_heading(float(body.heading))
        if body.guess_lat is not None:
            game.guess_lat = float(max(-85.0, min(85.0, body.guess_lat)))
        if body.guess_lon is not None:
            game.guess_lon = float(max(-180.0, min(180.0, body.guess_lon)))
        AGENT_STATE.game = game
        last_observation: dict[str, Any] = {
            "command": command,
            "received_at": datetime.now(UTC).isoformat(),
            "has_screenshot": bool(body.screenshot),
            "has_screenshot_url": bool(body.screenshot_url),
            "error": body.error,
        }
        AGENT_STATE.last_observation = last_observation
        if command == "capture" and body.screenshot:
            parsed = _parse_data_url_image(body.screenshot)
            if parsed:
                images = list(AGENT_STATE.captured_images or [])
                image_id = f"cap-{int(time.time() * 1000)}"
                saved_url = _persist_capture_image(image_id, parsed["mime"], parsed["data"])
                images.append(
                    {
                        "id": image_id,
                        "captured_at": datetime.now(UTC).isoformat(),
                        "mime": parsed["mime"],
                        "data": parsed["data"],
                        "screenshot_url": body.screenshot_url,
                        "saved_url": saved_url,
                    }
                )
                AGENT_STATE.captured_images = images[-8:]
                capture_debug = (
                        f"stored_data_url mime={parsed['mime']} bytes={len(parsed['data'])}"
                        + (f" saved={saved_url}" if saved_url else " saved=write_failed")
                )
            else:
                capture_debug = "data_url_invalid"
        elif command == "capture":
            raise NotImplementedError("This functionality is retired")
        if command == "capture":
            raise NotImplementedError("This functionality is retired")

    # Frontend already performs visual rotation and reports resulting POV.
    # Avoid applying additional backend-side rotate deltas which cause jumpy motion.
    if command in ("pan_random", "move_forward", "detect", "match", "guess"):
        await _apply_agent_action(command)
    await _mark_step_done(command)

    if command == "capture":
        await _mark_step_running("rotate_left")
        await _set_pending_command("rotate_left")
        await _agent_set_action("rotate_left")
    elif command == "rotate_left":
        await _mark_step_running("rotate_right")
        await _set_pending_command("rotate_right")
        await _agent_set_action("rotate_right")
    elif command == "rotate_right":
        await _mark_step_done("pan_random")
        await _mark_step_done("move_forward")
        await _mark_step_done("detect")
        await _mark_step_done("match")
        await _mark_step_running("guess")
        await _set_pending_command("guess")
        await _agent_set_action("guess")
    elif command == "guess":
        await _set_pending_command(None)
        async with AGENT_LOCK:
            AGENT_STATE.running = False

    return await _agent_snapshot()


@app.post("/api/game/start")
async def game_start():
    async with AGENT_LOCK:
        AGENT_STATE.game = _build_game_state()
        AGENT_STATE.error = None
        AGENT_STATE.running = False
        AGENT_STATE.pending_command = None
        AGENT_STATE.steps = []
        AGENT_STATE.last_observation = None
        AGENT_STATE.last_action = None
        AGENT_STATE.captured_images = []
    return await _agent_snapshot()


@app.get("/api/game/state")
async def game_state():
    async with AGENT_LOCK:
        game: GameState = GameState() if AGENT_STATE.game is None else AGENT_STATE.game
    return {"ok": bool(game), "game": asdict(game)}


@app.post("/api/game/action")
async def game_action(body: GameActionRequest):
    action = (body.action or "").strip().lower()
    valid = {step_id for step_id, _ in AGENT_STEP_SEQUENCE}
    if action not in valid:
        raise HTTPException(status_code=400, detail=f"unsupported action: {action}")
    if action == "guess" and (body.guess_lat is not None or body.guess_lon is not None):
        async with AGENT_LOCK:
            game: GameState = GameState() if AGENT_STATE.game is None else AGENT_STATE.game
            if body.guess_lat is not None:
                game.guess_lat = max(-85.0, min(85.0, body.guess_lat))
            if body.guess_lon is not None:
                game.guess_lon = max(-180.0, min(180.0, body.guess_lon))
            AGENT_STATE.game = game
    await _apply_agent_action(action)
    await _agent_set_action(action)
    return await _agent_snapshot()


@app.get("/api/agent/status")
async def agent_status():
    return await _agent_snapshot()


@app.get("/api/agent/frame")
async def agent_frame():
    _start_agent_thread()
    async with AGENT_LOCK:
        frame = AGENT_STATE.frame
        frame_mime = AGENT_STATE.frame_mime or "image/svg+xml"
        error = AGENT_STATE.error
        last_action = AGENT_STATE.last_action
        last_frame_at = AGENT_STATE.last_frame_at
        game: GameState = GameState() if AGENT_STATE.game is None else AGENT_STATE.game
    return {
        "ok": bool(frame),
        "frame": frame,
        "frame_mime": frame_mime,
        "error": error,
        "last_action": last_action,
        "last_frame_at": last_frame_at,
        "game": game,
    }
