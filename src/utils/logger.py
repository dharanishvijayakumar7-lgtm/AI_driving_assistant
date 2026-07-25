"""
logger.py — Centralized logging setup for the AI Driving Assistant.

Why this exists:
  Using Python's built-in `logging` module (instead of print statements) gives
  us timestamped, levelled, filterable output that can be routed to both the
  console and a file without changing any module-level code.

Usage in any module:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Source opened successfully.")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure the root logger for the entire application.

    Call this ONCE from main.py before any other module is imported.
    All subsequent `get_logger(__name__)` calls across the codebase will
    inherit this configuration automatically.

    Args:
        level:    One of "DEBUG", "INFO", "WARNING", "ERROR". Defaults to "INFO".
        log_file: Optional path to a file where logs should also be written.
                  If None, logs go to stdout only.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []

    # Console handler — always present
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # Optional file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a named logger.

    This is a thin convenience wrapper so every module gets a consistent
    logger without needing to import `logging` directly.

    Args:
        name: Typically passed as `__name__` so the logger hierarchy mirrors
              the package structure (e.g., "src.pipeline.video_source").

    Returns:
        A configured Logger instance.
    """
    return logging.getLogger(name)
