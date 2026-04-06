import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _normalize_api_key(raw: Optional[str], env_name: str) -> Optional[str]:
    """
    Normalize API key values from environment variables.

    Handles the common mistake where a value is pasted as
    "ENV_NAME=actual_key" instead of just "actual_key".
    """
    if raw is None:
        return None
    value = raw.strip().strip('"').strip("'")
    prefix = f"{env_name}="
    if value.startswith(prefix):
        value = value[len(prefix):].strip()
    return value or None

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("GGSolver")

_langsmith_tracing = os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes")
if _langsmith_tracing:
    _langsmith_project = os.getenv("LANGSMITH_PROJECT", "default")
    logging.getLogger("GGSolver").info(f"LangSmith tracing enabled (project: {_langsmith_project})")

GEMINI_API_KEY: Optional[str] = _normalize_api_key(os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY not found in .env")
GOOGLE_MAPS_API_KEY: Optional[str] = _normalize_api_key(os.getenv("GOOGLE_MAPS_API_KEY"), "GOOGLE_MAPS_API_KEY")
if GOOGLE_MAPS_API_KEY is None:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in .env")


def log_debug(s: str):
    logger.debug(s)


def log_event(message: str) -> None:
    logger.info(message)
    print(message, flush=True)
