import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("GGSolver")

GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY not found in .env")
GOOGLE_MAPS_API_KEY: Optional[str] = os.getenv("GOOGLE_MAPS_API_KEY")
if GOOGLE_MAPS_API_KEY is None:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in .env")


def log_debug(s: str):
    logger.debug(s)


def log_event(message: str) -> None:
    logger.info(message)
    print(message, flush=True)
