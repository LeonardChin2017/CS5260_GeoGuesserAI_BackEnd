import logging
import math
import os

from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("GGSolver")

def _0_if_nan(num: float) -> float:
    return 0.0 if math.isnan(num) else num


def log_debug(s: str):
    logger.debug(s)

def log_event(message: str) -> None:
    logger.info(message)
    print(message, flush=True)