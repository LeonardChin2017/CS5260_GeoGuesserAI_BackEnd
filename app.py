import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent import Agent, AnalysisResult
from game import Game

from pydantic import BaseModel

from util import log_event

AGENT_LOCK: asyncio.Lock = asyncio.Lock()
AGENT: Optional[Agent] = None

# When True, backend does NOT call real Google Maps or Gemini.
# Instead it serves fixed dummy data so the frontend can be developed
# without external dependencies or API costs.
DUMMY_MODE: bool = True

# Simple dummy-run state for frontend testing
DUMMY_RUNNING: bool = False
DUMMY_LOGS: list[dict[str, Any]] = []

# Tiny 1x1 white JPEG (base64, short) for placeholder frames.
_DUMMY_JPEG_BASE64: str = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
    "2wBDAf//////////////////////////////////////////////////////////////////////////////////////"
    "wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAb/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AMf/2Q=="
)

_DUMMY_GAME_SNAPSHOT: dict[str, Any] = {
    "view_lat": 15.48272989845598,
    "view_lon": -12.97018351821334,
    "heading": 0.0,
    "target_lat": 13.5,
    "target_lon": 5.5,
}

_DUMMY_ANALYSIS: dict[str, Any] = {
    "iter": 0,
    "game": _DUMMY_GAME_SNAPSHOT,
    "belief_state": [
        {
            "country": "Niger",
            "region": None,
            "lat": 13.5,
            "lon": 5.5,
            "confidence": 0.85,
            "evidence": "The semi-arid savanna biome, reddish-brown dirt roads, vernacular architecture, and the presence of Arabic script with West African language hints (Hausa, Fulfulde) strongly indicate Niger.",
        },
        {
            "country": "Mali",
            "region": None,
            "lat": 14.5,
            "lon": -2.0,
            "confidence": 0.75,
            "evidence": "Similar to Niger, Mali exhibits a semi-arid savanna climate, reddish soil, vernacular architecture, and uses Arabic script alongside local languages like Fulfulde.",
        },
        {
            "country": "Chad",
            "region": None,
            "lat": 12.5,
            "lon": 18.5,
            "confidence": 0.7,
            "evidence": "Chad shares the semi-arid savanna climate, reddish soil, and vernacular architecture, with Arabic being an official language, though the specific language hints are slightly less strong for Chad compared to West Africa.",
        },
    ],
    "action": {"type": "GUESS", "lat": 13.5, "lon": 5.5},
    "final_guess": {"lat": 13.5, "lon": 5.5, "country": "Niger", "confidence": 0.85},
    "specialist_outputs": {
        "text_language": {
            "agent": "text_language",
            "detected_scripts": ["Arabic"],
            "language_hints": ["Arabic", "Hausa", "Fulfulde", "Wolof"],
            "place_names": [],
            "confidence": 0.7,
            "evidence": "Arabic script is visible on the white wall of the building on the right side of the image, though the text is not clearly legible.",
        },
        "architecture": {
            "agent": "architecture",
            "building_styles": ["Vernacular", "Informal settlement architecture"],
            "materials": ["mud bricks", "concrete blocks", "corrugated metal"],
            "urban_density": "rural",
            "street_furniture": ["utility poles", "street light", "fences"],
            "national_flags_or_symbols": [],
            "confidence": 0.9,
            "evidence": "The simple, single-story structures built with unrendered mud bricks or concrete blocks, alongside unpaved roads, are characteristic of vernacular architecture in rural or informal settings.",
        },
        "road_infra": {
            "agent": "road_infra",
            "driving_side": "unknown",
            "road_markings": "No painted road markings are visible on the unpaved dirt road.",
            "sign_shapes_colors": "No standard road signs are visible.",
            "vehicle_types": [],
            "road_quality": "poor",
            "camera_rig_clues": [],
            "confidence": 0.9,
            "evidence": "The road is unpaved and made of dirt, indicating poor road quality.",
        },
        "climate_terrain": {
            "agent": "climate_terrain",
            "climate_zone": "semi-arid",
            "terrain_type": "flat_plain",
            "sky_and_light": "Clear, bright blue sky with a high sun angle, casting distinct and relatively short shadows, suggesting a low latitude and dry conditions.",
            "soil_color": "Reddish-brown to ochre.",
            "road_surface": "dirt",
            "confidence": 0.9,
            "evidence": "The reddish-brown dirt, sparse vegetation, and clear blue sky strongly indicate a semi-arid climate and flat plain terrain.",
        },
        "vegetation": {
            "agent": "vegetation",
            "vegetation_type": "tropical broadleaf",
            "biome": "savanna",
            "notable_species": ["broadleaf trees"],
            "season_hints": "trees are in leaf, indicating a growing season, but the ground is dry and dusty, suggesting a dry period or end of wet season.",
            "confidence": 0.9,
            "evidence": "The image displays scattered broadleaf trees amidst vast areas of dry, reddish soil, characteristic of a savanna biome.",
        },
    },
    "logs": [],
    "error": "",
}

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
    global AGENT, DUMMY_RUNNING, DUMMY_LOGS
    if DUMMY_MODE:
        # In dummy mode, simulate a fresh run with fixed logs.
        log_event("[dummy_start] Initializing dummy agent run")
        DUMMY_RUNNING = True
        DUMMY_LOGS = [
            {"iter": 0, "agent": "text_language", "evidence": "Dummy text_language evidence."},
            {"iter": 0, "agent": "architecture", "evidence": "Dummy architecture evidence."},
            {"iter": 0, "agent": "climate_terrain", "evidence": "Dummy climate_terrain evidence."},
            {"iter": 0, "agent": "vegetation", "evidence": "Dummy vegetation evidence."},
            {"iter": 0, "agent": "road_infra", "evidence": "Dummy road_infra evidence."},
        ]
        log_event(f"[dummy_start] Logs prepared: {len(DUMMY_LOGS)} entries")
        return {"ok": True, "running": True}
    async with AGENT_LOCK:
        if AGENT is None:
            AGENT = Agent()
    return {"ok": True}


@app.post("/api/agent/stop")
async def stop_agent():
    global AGENT, DUMMY_RUNNING, DUMMY_LOGS
    if DUMMY_MODE:
        log_event("[dummy_stop] Stopping dummy agent run and clearing logs")
        DUMMY_RUNNING = False
        DUMMY_LOGS = []
        return {"ok": True, "running": False}
    AGENT = None
    return {"ok": True, "running": False}


@app.get("/api/agent/status")
async def agent_status():
    global AGENT, DUMMY_RUNNING, DUMMY_LOGS
    if DUMMY_MODE:
        log_event(f"[dummy_status] running={DUMMY_RUNNING}, logs={len(DUMMY_LOGS)}")
        # If dummy agent is not running, expose a clean idle state with no result/steps.
        if not DUMMY_RUNNING:
            return {
                "running": False,
                "result": None,
                "logs": [],
                "steps": [],
                "game": _DUMMY_GAME_SNAPSHOT,
            }
        # Shape logs into 'steps' objects so the existing frontend
        # (which expects data.steps) can display them in the activity trace.
        steps: list[dict[str, Any]] = []
        for idx, entry in enumerate(DUMMY_LOGS):
            name = str(entry.get("agent", f"agent_{idx}"))
            title = f"{name.replace('_', ' ')} Agent"
            evidence = entry.get("evidence") or ""
            steps.append(
                {
                    "id": f"{entry.get('iter', 0)}-{name}-{idx}",
                    "message": f"{title}: {evidence}",
                    "status": "done",
                }
            )
        payload = {
            "running": DUMMY_RUNNING,
            "result": _DUMMY_ANALYSIS,
            "logs": DUMMY_LOGS,
            "steps": steps,
            "game": _DUMMY_GAME_SNAPSHOT,
        }
        # Log the exact JSON payload sent to the frontend for easier debugging.
        try:
            pretty = json.dumps(payload, indent=2, ensure_ascii=False)
            log_event(f"[dummy_status_payload]\n{pretty}")
        except Exception as exc:  # pragma: no cover - best-effort logging
            log_event(f"[dummy_status_payload] failed to serialize: {exc}")
        # Simulate an agent that auto-stops after producing a final result:
        # after we expose this snapshot once, mark as not running and clear
        # in-flight logs so the next status poll sees an idle state.
        DUMMY_RUNNING = False
        DUMMY_LOGS = []
        return payload
    return {"running": AGENT is not None}


@app.get("/api/agent/frame")
async def agent_frame():
    global AGENT
    out: dict[str, Any] = {}
    if DUMMY_MODE:
        log_event("[dummy_frame] Serving placeholder frame")
        out["frame"] = _DUMMY_JPEG_BASE64
        out["frame_mime"] = "image/jpeg"
        out["game"] = _DUMMY_GAME_SNAPSHOT
        out["last_action"] = "GUESS"
        out["last_frame_at"] = datetime.now(timezone.utc).isoformat()
        return out
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
    if DUMMY_MODE:
        return _DUMMY_ANALYSIS
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
    if DUMMY_MODE:
        return _DUMMY_ANALYSIS
    game: Game = Game()
    game.reset(req.start_lat, req.start_lon, req.start_heading)
    async with AGENT_LOCK:
        return AGENT.run(game, req.max_iter)