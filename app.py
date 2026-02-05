import base64
import json
import logging
import os
import re
import sqlite3
import time
import asyncio
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from playwright.async_api import async_playwright
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

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("jobai")


def log_debug(message: str) -> None:
    """Debug logging disabled (no-op)."""
    return

AGENT_LOCK = asyncio.Lock()
AGENT_STATE: Dict[str, Any] = {
    "running": False,
    "stop": False,
    "steps": [],
    "frame": None,
    "error": None,
    "last_action": None,
    "last_frame_at": None,
}
AGENT_QUEUE: "asyncio.Queue[str]" = asyncio.Queue()
AGENT_TASK_STARTED = False
AGENT_BROWSER: Dict[str, Any] = {
    "playwright": None,
    "browser": None,
    "page": None,
    "loaded": False,
}


async def _agent_snapshot() -> Dict[str, Any]:
    log_debug("agent_snapshot called")
    async with AGENT_LOCK:
        steps = [dict(step) for step in AGENT_STATE["steps"]]
        return {
            "running": bool(AGENT_STATE["running"]),
            "steps": steps,
            "error": AGENT_STATE.get("error"),
            "last_action": AGENT_STATE.get("last_action"),
            "last_frame_at": AGENT_STATE.get("last_frame_at"),
        }


async def _agent_set_steps(steps: List[Dict[str, Any]]) -> None:
    async with AGENT_LOCK:
        AGENT_STATE["steps"] = steps


async def _agent_set_frame(frame_b64: Optional[str]) -> None:
    async with AGENT_LOCK:
        AGENT_STATE["frame"] = frame_b64
        AGENT_STATE["last_frame_at"] = datetime.utcnow().isoformat()


async def _agent_set_error(message: Optional[str]) -> None:
    async with AGENT_LOCK:
        AGENT_STATE["error"] = message


async def _agent_set_action(action: Optional[str]) -> None:
    async with AGENT_LOCK:
        AGENT_STATE["last_action"] = action


async def _ensure_browser() -> None:
    if AGENT_BROWSER["page"] is not None:
        return
    log_debug("starting playwright browser")
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )
    page = await browser.new_page(viewport={"width": 1280, "height": 720})
    AGENT_BROWSER["playwright"] = playwright
    AGENT_BROWSER["browser"] = browser
    AGENT_BROWSER["page"] = page
    AGENT_BROWSER["loaded"] = False


async def _shutdown_browser() -> None:
    try:
        if AGENT_BROWSER["page"] is not None:
            await AGENT_BROWSER["page"].close()
    except Exception:
        pass
    try:
        if AGENT_BROWSER["browser"] is not None:
            await AGENT_BROWSER["browser"].close()
    except Exception:
        pass
    try:
        if AGENT_BROWSER["playwright"] is not None:
            await AGENT_BROWSER["playwright"].stop()
    except Exception:
        pass
    AGENT_BROWSER["page"] = None
    AGENT_BROWSER["browser"] = None
    AGENT_BROWSER["playwright"] = None
    AGENT_BROWSER["loaded"] = False


async def _ensure_page_loaded() -> None:
    log_debug("ensure_page_loaded called")
    await _ensure_browser()
    page = AGENT_BROWSER["page"]
    if page is None:
        return
    if AGENT_BROWSER["loaded"]:
        return
    log_debug("navigating to https://www.worldguessr.com/")
    await page.goto("https://www.worldguessr.com/", wait_until="domcontentloaded", timeout=60000)
    await asyncio.sleep(2.0)
    AGENT_BROWSER["loaded"] = True
    image_bytes = await page.screenshot(type="jpeg", quality=70)
    await _agent_set_frame(base64.b64encode(image_bytes).decode("utf-8"))


async def _agent_worker() -> None:
    log_debug("agent_worker started")
    min_delay = float(os.getenv("AGENT_STEP_DELAY_MIN_SECONDS", "5"))
    max_delay = float(os.getenv("AGENT_STEP_DELAY_MAX_SECONDS", "10"))
    sequence = [
        ("capture", "Capture current view"),
        ("rotate_left", "Rotate view left"),
        ("rotate_right", "Rotate view right"),
        ("pan_random", "Pan view randomly"),
        ("move_forward", "Move forward"),
        ("detect", "Detect signs, language, and road markings"),
        ("match", "Match visual clues with map patterns"),
        ("guess", "Place guess and submit"),
    ]
    steps_template = [{"id": sid, "message": msg, "status": "pending"} for sid, msg in sequence]
    current_step_index = -1
    step_started_at: Optional[float] = None
    current_step_delay: Optional[float] = None
    last_frame_at = 0.0
    frame_interval = float(os.getenv("AGENT_FRAME_INTERVAL_SECONDS", "0.6"))

    async def set_steps_for_start() -> None:
        nonlocal current_step_index, step_started_at
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
            await _shutdown_browser()
            try:
                await _ensure_page_loaded()
            except Exception as exc:
                await _agent_set_error(f"Browser init failed: {exc}")
                async with AGENT_LOCK:
                    AGENT_STATE["running"] = False
                continue
            async with AGENT_LOCK:
                AGENT_STATE["running"] = True
                AGENT_STATE["stop"] = False
            await set_steps_for_start()
        elif cmd == "stop":
            log_debug("agent_worker received STOP command")
            async with AGENT_LOCK:
                AGENT_STATE["running"] = False
                AGENT_STATE["stop"] = True
            current_step_index = -1
            step_started_at = None

        try:
            await _ensure_page_loaded()
        except Exception as exc:
            await _agent_set_error(f"Browser init failed: {exc}")
            log_debug(f"ensure_page_loaded failed: {exc}")
            await asyncio.sleep(1.0)
            continue

        page = AGENT_BROWSER["page"]
        if page is None:
            await asyncio.sleep(0.5)
            continue

        # Refresh frame periodically
        if time.monotonic() - last_frame_at > frame_interval:
            try:
                image_bytes = await page.screenshot(type="jpeg", quality=70)
                await _agent_set_frame(base64.b64encode(image_bytes).decode("utf-8"))
                last_frame_at = time.monotonic()
            except Exception as exc:
                await _agent_set_error(f"Frame capture failed: {exc}")
                log_debug(f"frame capture failed: {exc}")

        async with AGENT_LOCK:
            running = AGENT_STATE["running"]
            should_stop = AGENT_STATE["stop"]

        if running and not should_stop and current_step_index >= 0:
            if step_started_at is None or current_step_delay is None:
                step_started_at = time.monotonic()
                current_step_delay = random.uniform(min_delay, max_delay)
            elapsed = time.monotonic() - step_started_at
            if elapsed >= current_step_delay and current_step_index < len(sequence):
                # Perform action at step boundary
                sid, _ = sequence[current_step_index]
                log_debug(f"performing step {current_step_index} id={sid}")
                try:
                    if sid in ("rotate_left", "rotate_right"):
                        await _agent_set_action(sid)
                        drag = random.randint(140, 240)
                        direction = -drag if sid == "rotate_left" else drag
                        await page.mouse.move(640, 360)
                        await page.mouse.down()
                        await page.mouse.move(640 + direction, 360, steps=18)
                        await page.mouse.up()
                        await asyncio.sleep(0.8)
                    elif sid == "pan_random":
                        await _agent_set_action("pan_random")
                        dx = random.randint(-200, 200)
                        dy = random.randint(-120, 120)
                        await page.mouse.move(640, 360)
                        await page.mouse.down()
                        await page.mouse.move(640 + dx, 360 + dy, steps=16)
                        await page.mouse.up()
                        await asyncio.sleep(0.8)
                    elif sid == "move_forward":
                        await _agent_set_action("move_forward")
                        # Click near center to move forward on street view
                        cx = random.randint(520, 760)
                        cy = random.randint(300, 480)
                        await page.mouse.click(cx, cy)
                        await asyncio.sleep(1.2)
                    image_bytes = await page.screenshot(type="jpeg", quality=70)
                    await _agent_set_frame(base64.b64encode(image_bytes).decode("utf-8"))
                except Exception as exc:
                    await _agent_set_error(f"Action failed: {exc}")
                    log_debug(f"action {sid} failed: {exc}")
                async with AGENT_LOCK:
                    if AGENT_STATE["steps"]:
                        AGENT_STATE["steps"][current_step_index]["status"] = "done"
                        if current_step_index + 1 < len(AGENT_STATE["steps"]):
                            AGENT_STATE["steps"][current_step_index + 1]["status"] = "running"
                current_step_index += 1
                step_started_at = time.monotonic()
                current_step_delay = random.uniform(min_delay, max_delay)
                if current_step_index >= len(sequence):
                    async with AGENT_LOCK:
                        AGENT_STATE["running"] = False
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


def get_table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    try:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [row["name"] for row in cols]
    except Exception:
        return []


SYSTEM_PROMPT = read_prompt_file(
    "system.txt",
    "You are an exam question drafting assistant. Generate exam questions in plain text.",
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


def parse_style_update(message: str) -> Optional[Dict[str, int]]:
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


def build_question_list(formatted_text: str, user_message: str) -> Optional[List[str]]:
    lines = [l.strip() for l in formatted_text.split("\n") if l.strip()]
    numbered = [l for l in lines if re.match(r"^\d+\.\s+", l)]
    if not numbered:
        return None
    wants_single = re.search(r"\b(single|one)\s+(question|q)\b", user_message, re.I) is not None
    if len(numbered) == 1 and not wants_single:
        return None
    return [re.sub(r"^\d+\.\s+", "", l).strip() for l in numbered if l.strip()]


def generate_pdf(questions: List[str], title: str = "Generated Question Sheet") -> Optional[str]:
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


def call_gemini(contents: List[Dict[str, Any]], api_key: str, system_text: str) -> Optional[str]:
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


def call_deepseek(messages: List[Dict[str, str]], api_key: str, system_text: str) -> Optional[str]:
    url = "https://api.deepseek.com/v1/chat/completions"
    body = {"model": DEEPSEEK_MODEL, "messages": [{"role": "system", "content": system_text}] + messages}
    res = requests.post(url, json=body, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    if not res.ok:
        return None
    data = res.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or None


def build_contents(history: List[Dict[str, str]], message: str) -> List[Dict[str, Any]]:
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


class ChatRequest(BaseModel):
    message: str
    messages: Optional[List[Dict[str, str]]] = None
    apiKey: Optional[str] = None
    llmProvider: Optional[str] = None


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/generated_papers", StaticFiles(directory=GENERATED_PAPERS_DIR), name="generated_papers")


@app.get("/")
def root():
    return {"service": "jobai-backend", "message": "API only. Try GET /health or POST /api/chat."}


@app.get("/health")
def health():
    return {"ok": True}


@app.on_event("startup")
async def on_startup() -> None:
    _start_agent_thread()


@app.post("/api/agent/start")
async def start_agent():
    _start_agent_thread()
    async with AGENT_LOCK:
        if AGENT_STATE["running"]:
            return await _agent_snapshot()
        AGENT_STATE["stop"] = False
        AGENT_STATE["error"] = None
    await AGENT_QUEUE.put("start")
    return await _agent_snapshot()


@app.post("/api/agent/stop")
async def stop_agent():
    _start_agent_thread()
    async with AGENT_LOCK:
        AGENT_STATE["stop"] = True
        AGENT_STATE["running"] = False
    await AGENT_QUEUE.put("stop")
    return await _agent_snapshot()


@app.get("/api/agent/status")
async def agent_status():
    return await _agent_snapshot()


@app.get("/api/agent/frame")
async def agent_frame():
    _start_agent_thread()
    try:
        await _ensure_page_loaded()
    except Exception as exc:
        await _agent_set_error(f"Browser init failed: {exc}")
    async with AGENT_LOCK:
        frame = AGENT_STATE.get("frame")
        error = AGENT_STATE.get("error")
        last_action = AGENT_STATE.get("last_action")
        last_frame_at = AGENT_STATE.get("last_frame_at")
    return {
        "ok": bool(frame),
        "frame": frame,
        "error": error,
        "last_action": last_action,
        "last_frame_at": last_frame_at,
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
        "questionList": question_list,
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
def put_gemini_key(request: Request, body: Dict[str, Any]):
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
def put_deepseek_key(request: Request, body: Dict[str, Any]):
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
def put_llm_provider(request: Request, body: Dict[str, Any]):
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
def save_chat(request: Request, body: Dict[str, Any]):
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
def put_links(request: Request, body: Dict[str, Any]):
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
