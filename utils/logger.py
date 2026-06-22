"""
Centralized logging setup.

Import `get_logger(__name__)` from any module instead of using print().
Logs go to both the console and a rotating file under config.LOG_DIR.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

import config

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_logger() -> None:
    global _configured
    if _configured:
        return

    root_logger = logging.getLogger("meeting_assistant")
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_file = config.LOG_DIR / "app.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Returns a namespaced logger, e.g. get_logger(__name__) inside any
    module. All loggers nest under 'meeting_assistant' so a single root
    handler configuration covers everything.
    """
    _configure_root_logger()
    return logging.getLogger(f"meeting_assistant.{name}")
