import asyncio
import base64
import json
import logging
import os
import random
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "keys.db"
PROMPTS_DIR = BASE_DIR / "prompts"
GENERATED_PAPERS_DIR = BASE_DIR / "generated_papers"
GENERATED_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
CAPTURED_IMAGES_DIR = DATA_DIR / "captures"
CAPTURED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("jobai")
CAPTURE_PIPELINE_VERSION = "capture-v3-ext-sniff"


def log_debug(message: str) -> None:
    """Debug logging disabled (no-op)."""
    return

AGENT_LOCK = asyncio.Lock()


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
    game: Optional[dict] = None
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
        steps = [dict(step) for step in AGENT_STATE.steps]
        game = dict(AGENT_STATE.game) if isinstance(AGENT_STATE.game, dict) else None
        captured_images = [dict(img) for img in (AGENT_STATE.captured_images or [])]
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
    payload = raw[comma + 1 :]
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


def _wrap_heading(degrees: float) -> float:
    while degrees < 0:
        degrees += 360
    while degrees >= 360:
        degrees -= 360
    return degrees


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Fast equirectangular approximation is sufficient for game scoring.
    lat1r = lat1 * 0.017453292519943295
    lat2r = lat2 * 0.017453292519943295
    lon1r = lon1 * 0.017453292519943295
    lon2r = lon2 * 0.017453292519943295
    x = (lon2r - lon1r) * max(0.1, abs((lat1r + lat2r) / 2.0))
    y = lat2r - lat1r
    return (x * x + y * y) ** 0.5 * 6371.0


def _score_from_distance(distance_km: float) -> int:
    score = int(max(0.0, 5000.0 - (distance_km * 4.0)))
    return score


def _escape_svg_text(raw: str) -> str:
    return (
        raw.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _build_game_state() -> dict[str, Any]:
    target = random.choice(GEO_LOCATIONS)
    return {
        "target_name": target["name"],
        "target_country": target["country"],
        "target_lat": target["lat"],
        "target_lon": target["lon"],
        "clues": list(target["clues"]),
        "guess_lat": random.uniform(-60.0, 65.0),
        "guess_lon": random.uniform(-170.0, 170.0),
        "view_lat": target["lat"],
        "view_lon": target["lon"],
        "heading": random.uniform(0.0, 359.0),
        "confidence": 0.2,
        "detected_clues": [],
        "best_country_guess": None,
        "final_distance_km": None,
        "score": None,
    }


def _render_game_svg(game: dict[str, Any]) -> str:
    heading = int(float(game.get("heading") or 0))
    guess_lat = float(game.get("guess_lat") or 0.0)
    guess_lon = float(game.get("guess_lon") or 0.0)
    view_lat = float(game.get("view_lat") or 0.0)
    view_lon = float(game.get("view_lon") or 0.0)
    confidence = int(float(game.get("confidence") or 0.0) * 100)
    clues = game.get("detected_clues") or game.get("clues") or []
    visible_clues = clues[:4]
    distance = game.get("final_distance_km")
    score = game.get("score")
    best_country = game.get("best_country_guess") or "Analyzing..."
    result = (
        f"Distance: {distance:.0f} km | Score: {score}"
        if isinstance(distance, (int, float)) and isinstance(score, int)
        else "Round in progress"
    )
    clue_rows = "".join(
        f'<text x="40" y="{220 + i * 34}" font-size="20" fill="#d8e8ff">- {_escape_svg_text(str(clue))}</text>'
        for i, clue in enumerate(visible_clues)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="100%" stop-color="#1e3a8a"/>
    </linearGradient>
  </defs>
  <rect width="1280" height="720" fill="url(#bg)"/>
  <rect x="28" y="28" width="1224" height="664" rx="22" fill="#0b1220" stroke="#334155" stroke-width="2"/>
  <text x="40" y="76" font-size="32" fill="#f8fafc" font-family="Arial">GeoGuess AI Sandbox</text>
  <text x="40" y="118" font-size="20" fill="#93c5fd" font-family="Arial">Heading: {heading}° | View: ({view_lat:.4f}, {view_lon:.4f}) | Guess: ({guess_lat:.3f}, {guess_lon:.3f}) | Confidence: {confidence}%</text>
  <text x="40" y="170" font-size="24" fill="#facc15" font-family="Arial">Detected Clues</text>
  {clue_rows}
  <text x="40" y="620" font-size="22" fill="#a7f3d0" font-family="Arial">Best Country Match: {_escape_svg_text(str(best_country))}</text>
  <text x="40" y="660" font-size="24" fill="#fde68a" font-family="Arial">{_escape_svg_text(result)}</text>
</svg>"""
    return base64.b64encode(svg.encode("utf-8")).decode("ascii")


def _render_streetview_frame(game: dict[str, Any]) -> Optional[dict[str, str]]:
    if not GOOGLE_MAPS_API_KEY:
        return None
    lat = float(game.get("view_lat") if game.get("view_lat") is not None else game.get("target_lat") or 0.0)
    lon = float(game.get("view_lon") if game.get("view_lon") is not None else game.get("target_lon") or 0.0)
    heading = int(float(game.get("heading") or 0.0))
    params = {
        "size": "640x640",
        "scale": "2",
        "location": f"{lat:.6f},{lon:.6f}",
        "heading": str(heading),
        "pitch": "0",
        "fov": "90",
        "key": GOOGLE_MAPS_API_KEY,
    }
    try:
        res = requests.get("https://maps.googleapis.com/maps/api/streetview", params=params, timeout=20)
        if not res.ok:
            return None
        content_type = (res.headers.get("content-type") or "").split(";")[0].strip().lower()
        if not content_type.startswith("image/"):
            return None
        return {
            "frame": base64.b64encode(res.content).decode("ascii"),
            "mime": content_type,
        }
    except Exception:
        return None


def _render_streetview_from_view(
    lat: float,
    lon: float,
    heading: float,
    api_key_override: Optional[str] = None,
) -> Optional[dict[str, str]]:
    api_key = (api_key_override or GOOGLE_MAPS_API_KEY or "").strip()
    if not api_key:
        return None
    params = {
        "size": "640x640",
        "scale": "2",
        "location": f"{lat:.6f},{lon:.6f}",
        "heading": str(int(_wrap_heading(float(heading)))),
        "pitch": "0",
        "fov": "90",
        "key": api_key,
    }
    try:
        res = requests.get("https://maps.googleapis.com/maps/api/streetview", params=params, timeout=20)
        if not res.ok:
            return None
        content_type = (res.headers.get("content-type") or "").split(";")[0].strip().lower()
        if not content_type.startswith("image/"):
            return None
        return {
            "mime": content_type,
            "data": base64.b64encode(res.content).decode("ascii"),
        }
    except Exception:
        return None


def _render_streetview_from_url(url: Optional[str]) -> Optional[dict[str, str]]:
    raw_url = (url or "").strip()
    if not raw_url:
        return None
    if not raw_url.startswith("https://maps.googleapis.com/maps/api/streetview"):
        return None
    try:
        res = requests.get(raw_url, timeout=20)
        if not res.ok:
            return None
        content_type = (res.headers.get("content-type") or "").split(";")[0].strip().lower()
        if not content_type.startswith("image/"):
            return None
        return {
            "mime": content_type,
            "data": base64.b64encode(res.content).decode("ascii"),
        }
    except Exception:
        return None


async def _agent_refresh_frame() -> None:
    async with AGENT_LOCK:
        game = dict(AGENT_STATE.game or {})
    if not game:
        return
    streetview = _render_streetview_frame(game)
    if streetview:
        await _agent_set_frame(streetview["frame"], streetview["mime"])
    else:
        await _agent_set_frame(_render_game_svg(game), "image/svg+xml")


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
        game = dict(AGENT_STATE.game or {})
    if not game:
        return
    if step_id == "rotate_left":
        game["heading"] = _wrap_heading(float(game["heading"]) - random.uniform(25.0, 70.0))
    elif step_id == "rotate_right":
        game["heading"] = _wrap_heading(float(game["heading"]) + random.uniform(25.0, 70.0))
    elif step_id == "pan_random":
        game["view_lon"] = max(-180.0, min(180.0, float(game["view_lon"]) + random.uniform(-0.002, 0.002)))
        game["view_lat"] = max(-85.0, min(85.0, float(game["view_lat"]) + random.uniform(-0.001, 0.001)))
        game["guess_lon"] = max(-180.0, min(180.0, float(game["guess_lon"]) + random.uniform(-16.0, 16.0)))
        game["guess_lat"] = max(-85.0, min(85.0, float(game["guess_lat"]) + random.uniform(-8.0, 8.0)))
    elif step_id == "move_forward":
        heading = float(game["heading"])
        lat_step = 0.8 * (1 if 0 <= heading < 180 else -1)
        lon_step = 1.2 * (1 if heading <= 90 or heading >= 270 else -1)
        game["view_lat"] = max(-85.0, min(85.0, float(game["view_lat"]) + (lat_step * 0.001)))
        game["view_lon"] = max(-180.0, min(180.0, float(game["view_lon"]) + (lon_step * 0.001)))
        game["guess_lat"] = max(-85.0, min(85.0, float(game["guess_lat"]) + lat_step))
        game["guess_lon"] = max(-180.0, min(180.0, float(game["guess_lon"]) + lon_step))
        game["confidence"] = min(0.95, float(game["confidence"]) + 0.08)
    elif step_id == "detect":
        clues = list(game.get("clues") or [])
        seen = list(game.get("detected_clues") or [])
        if len(seen) < len(clues):
            seen.append(clues[len(seen)])
        game["detected_clues"] = seen
        game["confidence"] = min(0.98, float(game["confidence"]) + 0.18)
    elif step_id == "match":
        game["best_country_guess"] = game.get("target_country")
        game["confidence"] = min(0.99, float(game["confidence"]) + 0.15)
    elif step_id == "guess":
        distance = _distance_km(
            float(game["guess_lat"]),
            float(game["guess_lon"]),
            float(game["target_lat"]),
            float(game["target_lon"]),
        )
        game["final_distance_km"] = distance
        game["score"] = _score_from_distance(distance)
    async with AGENT_LOCK:
        AGENT_STATE.game = game
    await _agent_refresh_frame()


async def _agent_worker() -> None:
    log_debug("agent_worker started")
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
            log_debug("agent_worker received START command")
            await _agent_set_error(None)
            async with AGENT_LOCK:
                AGENT_STATE.game = _build_game_state()
            async with AGENT_LOCK:
                AGENT_STATE.running = True
                AGENT_STATE.stop = False
            await set_steps_for_start()
            await _agent_refresh_frame()
        elif cmd == "stop":
            log_debug("agent_worker received STOP command")
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
                log_debug(f"performing step {current_step_index} id={sid}")
                try:
                    await _agent_set_action(sid)
                    await _apply_agent_action(sid)
                except Exception as exc:
                    await _agent_set_error(f"Action failed: {exc}")
                    log_debug(f"action {sid} failed: {exc}")
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

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_db() -> None:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_gemini_keys (
          user_id TEXT PRIMARY KEY,
          api_key TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_deepseek_keys (
          user_id TEXT PRIMARY KEY,
          api_key TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_llm_preference (
          user_id TEXT PRIMARY KEY,
          provider TEXT,
          updated_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_chat (
          user_id TEXT PRIMARY KEY,
          messages TEXT,
          updated_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_links (
          user_id TEXT PRIMARY KEY,
          links TEXT
        )
        """
    )
    ensure_column(conn, "user_gemini_keys", "api_key", "TEXT")
    ensure_column(conn, "user_deepseek_keys", "api_key", "TEXT")
    ensure_column(conn, "user_llm_preference", "provider", "TEXT")
    ensure_column(conn, "user_llm_preference", "updated_at", "TEXT")
    ensure_column(conn, "user_chat", "updated_at", "TEXT")
    ensure_column(conn, "user_links", "links", "TEXT")
    now = datetime.utcnow().isoformat()
    try:
        conn.execute("UPDATE user_llm_preference SET updated_at = ? WHERE updated_at IS NULL", (now,))
    except Exception:
        pass
    conn.commit()
    conn.close()


def read_prompt_file(file_name: str, fallback: str) -> str:
    try:
        content = (PROMPTS_DIR / file_name).read_text(encoding="utf-8")
        cleaned = content.replace("\r\n", "\n").strip()
        return cleaned or fallback
    except Exception:
        return fallback


def ensure_column(conn: sqlite3.Connection, table: str, column: str, col_def: str) -> None:
    try:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        if any(row["name"] == column for row in cols):
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")
    except Exception:
        return


def get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [row["name"] for row in cols]
    except Exception:
        return []


SYSTEM_PROMPT = read_prompt_file(
    "system.txt",
    "You are an exam question drafting assistant. Generate exam questions in plain text.", # TODO overwrite
)

FORMAT_GUARD_PROMPT = read_prompt_file(
    "format_guard.txt",
    "Return only questions, one per line, numbered 1., 2., 3.",
)


def _decode_jwt_subject(token: str) -> Optional[str]:
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    pad = "=" * (-len(payload) % 4)
    try:
        data = base64.urlsafe_b64decode(payload + pad)
        parsed = json.loads(data.decode("utf-8"))
        sub = parsed.get("sub")
        return sub if isinstance(sub, str) and sub.strip() else None
    except Exception:
        return None


def get_user_id(request: Request) -> str:
    auth = request.headers.get("authorization", "").strip()
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        sub = _decode_jwt_subject(token)
        return sub or (token or "anonymous")
    return auth or "anonymous"


def get_user_api_key(conn: sqlite3.Connection, user_id: str, provider: str) -> str:
    try:
        if provider == "deepseek":
            row = conn.execute(
                "SELECT api_key FROM user_deepseek_keys WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return (row["api_key"] or "").strip() if row else ""
        columns = set(get_table_columns(conn, "user_gemini_keys"))
        if "api_key" in columns:
            row = conn.execute(
                "SELECT api_key FROM user_gemini_keys WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row and row["api_key"]:
                return str(row["api_key"]).strip()
        if "encrypted_key" in columns:
            row = conn.execute(
                "SELECT encrypted_key FROM user_gemini_keys WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return (row["encrypted_key"] or "").strip() if row else ""
    except Exception:
        return ""
    return ""


def mask_key(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}***{value[-3:]}"


def log_event(message: str) -> None:
    logger.info(message)
    print(message, flush=True)


def parse_style_update(message: str) -> Optional[dict[str, int]]:
    text = message.lower()
    if not re.search(r"(font|text).*(size|bigger|smaller|larger)|increase.*font|decrease.*font", text):
        return None
    if re.search(r"(bigger|larger|increase)", text):
        return {"fontSizeDelta": 2}
    if re.search(r"(smaller|decrease|reduce)", text):
        return {"fontSizeDelta": -2}
    return None


def strip_markdown(text: str) -> str:
    return re.sub(r"^#+\s*", "", re.sub(r"\*([^*]+)\*", r"\1", text, flags=re.M)).strip()


def build_question_list(formatted_text: str, user_message: str) -> Optional[list[str]]:
    lines = [l.strip() for l in formatted_text.split("\n") if l.strip()]
    numbered = [l for l in lines if re.match(r"^\d+\.\s+", l)]
    if not numbered:
        return None
    wants_single = re.search(r"\b(single|one)\s+(question|q)\b", user_message, re.I) is not None
    if len(numbered) == 1 and not wants_single:
        return None
    return [re.sub(r"^\d+\.\s+", "", l).strip() for l in numbered if l.strip()]


def generate_pdf(questions: list[str], title: str = "Generated Question Sheet") -> Optional[str]:
    if not questions:
        return None
    file_name = f"paper_{int(datetime.utcnow().timestamp() * 1000)}.pdf"
    file_path = GENERATED_PAPERS_DIR / file_name
    try:
        c = canvas.Canvas(str(file_path), pagesize=A4)
        width, height = A4
        y = height - 72
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width / 2, y, title)
        y -= 24
        c.setFont("Helvetica", 9)
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawCentredString(width / 2, y, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        y -= 36
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 12)
        for idx, q in enumerate(questions, start=1):
            if y < 72:
                c.showPage()
                y = height - 72
                c.setFont("Helvetica", 12)
            c.drawString(72, y, f"{idx}. {q}")
            y -= 20
        c.save()
        return file_name
    except Exception:
        return None


def call_gemini(contents: list[dict[str, Any]], api_key: str, system_text: str) -> Optional[str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    body = {"systemInstruction": {"parts": [{"text": system_text}]}, "contents": contents}
    res = requests.post(url, json=body, timeout=60)
    if not res.ok:
        return None
    data = res.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    for part in parts:
        if "text" in part and part["text"]:
            return part["text"].strip()
    return None


def call_deepseek(messages: list[dict[str, str]], api_key: str, system_text: str) -> Optional[str]:
    url = "https://api.deepseek.com/v1/chat/completions"
    body = {"model": DEEPSEEK_MODEL, "messages": [{"role": "system", "content": system_text}] + messages}
    res = requests.post(url, json=body, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    if not res.ok:
        return None
    data = res.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or None


def build_contents(history: list[dict[str, str]], message: str) -> list[dict[str, Any]]:
    legacy = re.compile(r"\b(jobai|job search|job listings|application status|resume)\b", re.I)
    if any(legacy.search(m.get("content", "")) for m in history):
        history = []
    parts = []
    for m in history:
        role = "user" if m.get("role") == "user" else "model"
        text = m.get("content", "").strip()
        if text:
            parts.append({"role": role, "parts": [{"text": text}]})
    parts.append({"role": "user", "parts": [{"text": message.strip()}]})
    return parts

@dataclass
class ChatRequest(BaseModel):
    message: str = ''
    messages: Optional[list[dict[str, str]]] = None
    apiKey: Optional[str] = None
    llmProvider: Optional[str] = None

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/generated_papers", StaticFiles(directory=GENERATED_PAPERS_DIR), name="generated_papers")
app.mount("/captured_images", StaticFiles(directory=CAPTURED_IMAGES_DIR), name="captured_images")


@app.get("/")
def root():
    return {"service": "jobai-backend", "message": "API only. Try GET /health or POST /api/chat."}


@app.get("/health")
def health():
    return {"ok": True}


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


@app.post("/api/agent/start")
async def start_agent():
    async with AGENT_LOCK:
        if AGENT_STATE.running:
            return await _agent_snapshot()
        AGENT_STATE.running = True
        AGENT_STATE.stop = False
        AGENT_STATE.error = None
        AGENT_STATE.game = _build_game_state()
        AGENT_STATE.steps = [{"id": sid, "message": msg, "status": "pending"} for sid, msg in AGENT_STEP_SEQUENCE]
        AGENT_STATE.last_observation = None
        AGENT_STATE.captured_images = []
    await _mark_step_running("capture")
    await _set_pending_command("capture")
    await _agent_set_action("capture")
    await _agent_refresh_frame()
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
        game = dict(AGENT_STATE.game or {})
        if body.view_lat is not None:
            game["view_lat"] = float(max(-85.0, min(85.0, body.view_lat)))
        if body.view_lon is not None:
            game["view_lon"] = float(max(-180.0, min(180.0, body.view_lon)))
        if body.heading is not None:
            game["heading"] = _wrap_heading(float(body.heading))
        if body.guess_lat is not None:
            game["guess_lat"] = float(max(-85.0, min(85.0, body.guess_lat)))
        if body.guess_lon is not None:
            game["guess_lon"] = float(max(-180.0, min(180.0, body.guess_lon)))
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
                        "captured_at": datetime.utcnow().isoformat(),
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
            captured = None
            if body.screenshot_url:
                captured = _render_streetview_from_url(body.screenshot_url)
            if body.view_lat is not None and body.view_lon is not None:
                captured = captured or _render_streetview_from_view(
                    float(body.view_lat),
                    float(body.view_lon),
                    float(body.heading or 0.0),
                    body.maps_api_key,
                )
            if captured:
                images = list(AGENT_STATE.captured_images or [])
                image_id = f"cap-{int(time.time() * 1000)}"
                saved_url = _persist_capture_image(image_id, captured["mime"], captured["data"])
                images.append(
                    {
                        "id": image_id,
                        "captured_at": datetime.utcnow().isoformat(),
                        "mime": captured["mime"],
                        "data": captured["data"],
                        "screenshot_url": None,
                        "saved_url": saved_url,
                    }
                )
                AGENT_STATE.captured_images = images[-8:]
                capture_debug = (
                    f"stored_backend_view mime={captured['mime']} bytes={len(captured['data'])}"
                    + (f" saved={saved_url}" if saved_url else " saved=write_failed")
                )
            else:
                capture_debug = "missing_frontend_screenshot_and_backend_view_failed"
        if command == "capture":
            AGENT_STATE.last_observation["capture_debug"] = capture_debug

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

    await _agent_refresh_frame()
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
    await _agent_refresh_frame()
    return await _agent_snapshot()


@app.get("/api/game/state")
async def game_state():
    async with AGENT_LOCK:
        game = dict(AGENT_STATE.game or {})
    return {"ok": bool(game), "game": game}


@app.post("/api/game/action")
async def game_action(body: GameActionRequest):
    action = (body.action or "").strip().lower()
    valid = {step_id for step_id, _ in AGENT_STEP_SEQUENCE}
    if action not in valid:
        raise HTTPException(status_code=400, detail=f"unsupported action: {action}")
    if action == "guess" and (body.guess_lat is not None or body.guess_lon is not None):
        async with AGENT_LOCK:
            game = dict(AGENT_STATE.game or {})
            if body.guess_lat is not None:
                game["guess_lat"] = float(max(-85.0, min(85.0, body.guess_lat)))
            if body.guess_lon is not None:
                game["guess_lon"] = float(max(-180.0, min(180.0, body.guess_lon)))
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
        has_frame = bool(AGENT_STATE.frame)
    if not has_frame:
        await _agent_refresh_frame()
    async with AGENT_LOCK:
        frame = AGENT_STATE.frame
        frame_mime = AGENT_STATE.frame_mime or "image/svg+xml"
        error = AGENT_STATE.error
        last_action = AGENT_STATE.last_action
        last_frame_at = AGENT_STATE.last_frame_at
        game = dict(AGENT_STATE.game or {})
    return {
        "ok": bool(frame),
        "frame": frame,
        "frame_mime": frame_mime,
        "error": error,
        "last_action": last_action,
        "last_frame_at": last_frame_at,
        "game": game,
    }




@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message required")

    history = req.messages or []
    provider = "deepseek" if (req.llmProvider or "").lower() == "deepseek" else "gemini"
    user_id = get_user_id(request)
    log_event(f"chat request user={user_id} provider={provider}")

    api_key = (req.apiKey or "").strip()
    if not api_key:
        try:
            conn = get_db()
            api_key = get_user_api_key(conn, user_id, provider)
            conn.close()
        except Exception:
            api_key = ""
    source = "request" if api_key else "none"
    if not api_key:
        env_key = os.getenv("DEEPSEEK_API_KEY" if provider == "deepseek" else "GEMINI_API_KEY", "").strip()
        if env_key:
            api_key = env_key
            source = "env"
        else:
            source = "none"
    log_event(f"chat key source={source} key={mask_key(api_key)}")

    if not api_key:
        return JSONResponse(
            status_code=200,
            content={"reply": "I couldn't generate a response. Please add your API key in Settings and try again."},
        )

    contents = build_contents(history, message)
    if provider == "deepseek":
        messages = []
        for c in contents:
            role = "assistant" if c["role"] == "model" else "user"
            text = c["parts"][0]["text"]
            messages.append({"role": role, "content": text})
        reply = call_deepseek(messages, api_key, SYSTEM_PROMPT)
    else:
        reply = call_gemini(contents, api_key, SYSTEM_PROMPT)

    if not reply:
        return {"reply": "No response"}

    reply = strip_markdown(reply)
    style_update = parse_style_update(message)

    # Format-guard pass
    if provider == "deepseek":
        guard_reply = call_deepseek([{"role": "user", "content": reply}], api_key, FORMAT_GUARD_PROMPT)
    else:
        guard_reply = call_gemini([{"role": "user", "parts": [{"text": reply}]}], api_key, FORMAT_GUARD_PROMPT)

    if guard_reply:
        reply = strip_markdown(guard_reply)

    question_list = build_question_list(reply, message)
    pdf_url = None
    if question_list:
        file_name = generate_pdf(question_list)
        if file_name:
            base = f"{request.url.scheme}://{request.url.hostname}:{request.url.port}"
            pdf_url = f"{base}/generated_papers/{file_name}"

    return {
        "reply": reply,
        "questionlist": question_list,
        "styleUpdate": style_update,
        "pdfUrl": pdf_url,
    }


@app.get("/api/user/gemini-key")
def get_gemini_key(request: Request):
    user_id = get_user_id(request)
    conn = get_db()
    columns = set(get_table_columns(conn, "user_gemini_keys"))
    if "api_key" in columns:
        row = conn.execute(
            "SELECT api_key FROM user_gemini_keys WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        value = row["api_key"] if row else None
    elif "encrypted_key" in columns:
        row = conn.execute(
            "SELECT encrypted_key FROM user_gemini_keys WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        value = row["encrypted_key"] if row else None
    else:
        value = None
    conn.close()
    log_event(f"get_gemini_key user={user_id} hasKey={bool(value)} cols={sorted(columns)}")
    return {"hasKey": bool(value)}


@app.put("/api/user/gemini-key")
def put_gemini_key(request: Request, body: dict[str, Any]):
    user_id = get_user_id(request)
    api_key = (body.get("apiKey") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="apiKey required")
    conn = get_db()
    columns = set(get_table_columns(conn, "user_gemini_keys"))
    log_event(f"put_gemini_key user={user_id} cols={sorted(columns)} key={mask_key(api_key)}")
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError:
        pass
    last_error: Optional[Exception] = None
    for _ in range(5):
        try:
            if "encrypted_key" in columns and "api_key" in columns:
                conn.execute(
                    "INSERT INTO user_gemini_keys (user_id, api_key, encrypted_key) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(user_id) DO UPDATE SET api_key = ?, encrypted_key = ?",
                    (user_id, api_key, api_key, api_key, api_key),
                )
            elif "encrypted_key" in columns:
                conn.execute(
                    "INSERT INTO user_gemini_keys (user_id, encrypted_key) "
                    "VALUES (?, ?) "
                    "ON CONFLICT(user_id) DO UPDATE SET encrypted_key = ?",
                    (user_id, api_key, api_key),
                )
            else:
                conn.execute(
                    "INSERT INTO user_gemini_keys (user_id, api_key) "
                    "VALUES (?, ?) "
                    "ON CONFLICT(user_id) DO UPDATE SET api_key = ?",
                    (user_id, api_key, api_key),
                )
            conn.commit()
            conn.close()
            return {"ok": True}
        except sqlite3.OperationalError as exc:
            last_error = exc
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            time.sleep(0.1)
    conn.close()
    raise HTTPException(status_code=503, detail=f"database is locked: {last_error}")


@app.delete("/api/user/gemini-key")
def delete_gemini_key(request: Request):
    user_id = get_user_id(request)
    conn = get_db()
    log_event(f"delete_gemini_key user={user_id}")
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError:
        pass
    last_error: Optional[Exception] = None
    for _ in range(5):
        try:
            conn.execute("DELETE FROM user_gemini_keys WHERE user_id = ?", (user_id,))
            conn.commit()
            conn.close()
            return {"ok": True}
        except sqlite3.OperationalError as exc:
            last_error = exc
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            time.sleep(0.1)
    conn.close()
    raise HTTPException(status_code=503, detail=f"database is locked: {last_error}")


@app.get("/api/user/deepseek-key")
def get_deepseek_key(request: Request):
    user_id = get_user_id(request)
    conn = get_db()
    row = conn.execute("SELECT api_key FROM user_deepseek_keys WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return {"hasKey": bool(row and row["api_key"])}


@app.put("/api/user/deepseek-key")
def put_deepseek_key(request: Request, body: dict[str, Any]):
    user_id = get_user_id(request)
    api_key = (body.get("apiKey") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="apiKey required")
    conn = get_db()
    conn.execute(
        "INSERT INTO user_deepseek_keys (user_id, api_key) VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET api_key = ?",
        (user_id, api_key, api_key),
    )
    conn.commit()
    conn.close()
    return {"ok": True}


@app.delete("/api/user/deepseek-key")
def delete_deepseek_key(request: Request):
    user_id = get_user_id(request)
    conn = get_db()
    conn.execute("DELETE FROM user_deepseek_keys WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/api/user/llm-provider")
def get_llm_provider(request: Request):
    user_id = get_user_id(request)
    conn = get_db()
    row = conn.execute("SELECT provider FROM user_llm_preference WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return {"provider": row["provider"] if row else "gemini"}


@app.put("/api/user/llm-provider")
def put_llm_provider(request: Request, body: dict[str, Any]):
    user_id = get_user_id(request)
    provider = (body.get("provider") or "gemini").strip()
    if provider not in ("gemini", "deepseek"):
        provider = "gemini"
    conn = get_db()
    now = datetime.utcnow().isoformat()
    try:
        conn.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError:
        pass
    last_error: Optional[Exception] = None
    for _ in range(5):
        try:
            conn.execute(
                "INSERT INTO user_llm_preference (user_id, provider, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET provider = ?, updated_at = ?",
                (user_id, provider, now, provider, now),
            )
            conn.commit()
            conn.close()
            return {"ok": True}
        except sqlite3.OperationalError as exc:
            last_error = exc
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            time.sleep(0.1)
    conn.close()
    raise HTTPException(status_code=503, detail=f"database is locked: {last_error}")


@app.get("/api/user/chat")
def get_chat(request: Request):
    user_id = get_user_id(request)
    conn = get_db()
    row = conn.execute("SELECT messages FROM user_chat WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    if not row or not row["messages"]:
        return {"messages": []}
    try:
        return {"messages": json.loads(row["messages"])}
    except Exception:
        return {"messages": []}


@app.post("/api/user/chat")
def save_chat(request: Request, body: dict[str, Any]):
    user_id = get_user_id(request)
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages array required")
    conn = get_db()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO user_chat (user_id, messages, updated_at) VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET messages = ?, updated_at = ?",
        (user_id, json.dumps(messages), now, json.dumps(messages), now),
    )
    conn.commit()
    conn.close()
    return {"ok": True}


@app.delete("/api/user/chat")
def clear_chat(request: Request):
    user_id = get_user_id(request)
    conn = get_db()
    conn.execute("DELETE FROM user_chat WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/api/user/links")
def get_links(request: Request):
    user_id = get_user_id(request)
    conn = get_db()
    row = conn.execute("SELECT links FROM user_links WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    if not row or not row["links"]:
        return {"links": []}
    try:
        return {"links": json.loads(row["links"])}
    except Exception:
        return {"links": []}


@app.put("/api/user/links")
def put_links(request: Request, body: dict[str, Any]):
    user_id = get_user_id(request)
    links = body.get("links")
    if not isinstance(links, list):
        raise HTTPException(status_code=400, detail="links array required")
    conn = get_db()
    conn.execute(
        "INSERT INTO user_links (user_id, links) VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET links = ?",
        (user_id, json.dumps(links), json.dumps(links)),
    )
    conn.commit()
    conn.close()
    return {"ok": True}


init_db()
