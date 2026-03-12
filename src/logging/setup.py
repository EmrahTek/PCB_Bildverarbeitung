# logging config (RotatingFileHandler)
#Einheitliches Logging für die gesamte App (Console + rotierende Datei), via YAML oder dictConfig.
"""
setup.py

This module provides centralized logging setup for the application.
It is responsible for:
- configuring console/file handlers
- enabling log rotation
- applying a consistent log format across modules

Inputs:
- Path to a logging configuration YAML (or dictConfig structure).

Outputs:
- Global logging configuration applied via logging.config.dictConfig
- Helper function to retrieve module-level loggers

logging basics:
https://docs.python.org/3/library/logging.html

logging.config.dictConfig:
https://docs.python.org/3/library/logging.config.html

RotatingFileHandler:
https://docs.python.org/3/library/logging.handlers.html


Zu implementierende Funktionen

setup_logging(logging_config_path: Path) -> None

get_logger(name: str) -> logging.Logger

(Optional) log_environment_info(logger) -> None (Python-Version, OS, OpenCV-Version)


"""
from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any

from src.utils.io import load_yaml


def setup_logging(logging_config_path: str | Path, *, default_level: int = logging.INFO) -> None:
    """Configure logging from YAML with a safe fallback."""
    path = Path(logging_config_path)
    if not path.exists():
        logging.basicConfig(level=default_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        logging.getLogger(__name__).warning("Logging config not found: %s", path)
        return

    try:
        config = load_yaml(path)
        _ensure_log_dirs(config)
        logging.config.dictConfig(config)
        logging.getLogger(__name__).info("Logging configured from %s", path)
    except Exception as exc:
        logging.basicConfig(level=default_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        logging.getLogger(__name__).exception("Failed to configure logging from %s: %s", path, exc)


def _ensure_log_dirs(config: dict[str, Any]) -> None:
    handlers = config.get("handlers", {})
    if not isinstance(handlers, dict):
        return
    for handler in handlers.values():
        if isinstance(handler, dict) and "filename" in handler:
            Path(str(handler["filename"])).parent.mkdir(parents=True, exist_ok=True)

