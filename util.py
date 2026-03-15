import logging
import os

from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("GGSolver")


def log_debug(s: str):
    logger.debug(s)


def log_event(message: str) -> None:
    logger.info(message)
    print(message, flush=True)
