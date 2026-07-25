"""
config.py — Load and validate runtime configuration from config.yaml.

Why this exists:
  Centralizing config loading and validation here means every other module
  can receive a plain Python dict and trust it has been sanitized. If a
  required key is missing or has an invalid value, we fail fast at startup
  with a clear error rather than a cryptic KeyError deep in the pipeline.
"""

from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Keys that MUST be present in the loaded config
_REQUIRED_KEYS = [
    ("source", "type"),
    ("display", "width"),
    ("display", "height"),
    ("display", "window_title"),
    ("output", "save_video"),
    ("logging", "level"),
]

_VALID_SOURCE_TYPES = {"file", "webcam"}


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """
    Load config.yaml, validate required keys, and return a nested dict.

    Args:
        config_path: Path to the YAML config file, relative to the project root
                     or absolute.

    Returns:
        A validated configuration dictionary mirroring the YAML structure.

    Raises:
        FileNotFoundError: If the YAML file does not exist at the given path.
        ValueError:        If required keys are missing or contain invalid values.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: '{path.resolve()}'. "
            "Make sure you are running from the project root directory."
        )

    with path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    logger.debug("Raw config loaded from '%s'.", path)

    _validate(config)

    logger.info("Configuration loaded and validated from '%s'.", path)
    return config


def _validate(config: dict[str, Any]) -> None:
    """
    Validate that required keys exist and have sensible values.

    Args:
        config: The raw config dict loaded from YAML.

    Raises:
        ValueError: On any validation failure.
    """
    # Check required key paths
    for key_path in _REQUIRED_KEYS:
        node = config
        for key in key_path:
            if not isinstance(node, dict) or key not in node:
                raise ValueError(
                    f"Missing required config key: {' -> '.join(key_path)}"
                )
            node = node[key]

    # Validate source type
    source_type = config["source"]["type"]
    if source_type not in _VALID_SOURCE_TYPES:
        raise ValueError(
            f"Invalid source.type '{source_type}'. "
            f"Must be one of: {_VALID_SOURCE_TYPES}"
        )

    # Validate source-specific keys
    if source_type == "file":
        if not config["source"].get("file_path"):
            raise ValueError(
                "source.file_path must be set when source.type is 'file'."
            )
    elif source_type == "webcam":
        if config["source"].get("webcam_index") is None:
            raise ValueError(
                "source.webcam_index must be set when source.type is 'webcam'."
            )

    # Validate display dimensions
    for dim_key in ("width", "height"):
        val = config["display"].get(dim_key)
        if val is not None and (not isinstance(val, int) or val <= 0):
            raise ValueError(
                f"display.{dim_key} must be a positive integer or null, got: {val!r}"
            )
