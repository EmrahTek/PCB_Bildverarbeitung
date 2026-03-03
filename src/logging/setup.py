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
from typing import Any, Dict

def setup_logging(logging_config_path: str | Path, *, default_level: int = logging.INFO):
    """
    Setup logging from a YAML file (dictConfig), with a safe fallback.

    Args:
        logging_config_path: Path to config/logging.yaml
        default_level: Used if config cannot be loaded.
    """
    path = Path(logging_config_path)

    if not path.exists():
        logging.basicConfig(level=default_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logging.getLogger(__name__).warning("Logging config not found: %s (using basicConfig)",path)
        return
    try: 
        config = _load_yaml(path)
        _ensure_log_dirs(config)
        logging.config.dictConfig(config)
        logging.getLogger(__name__).info("Logging configured from %s", path)
    except Exception as exc:
        logging.basicConfig(level=default_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logging.getLogger(__name__).exception("Failed to load logging config (fallback basicConfig). Error: %s", exc)




def _load_yaml(path: Path) -> Dict[str,Any]: # binde stricht bedeutet private funktion. 
    """
    Load YAML file into a dict.
    Kept local to avoid scattering YAML dependency across the project.
    """
    try: 
        import yaml # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is not installed. Install it or remove YAML logging config usage.") from exc
    
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data,dict):
        raise ValueError("Logging YAML must define a dict at top level.")
    return data


def _ensure_log_dirs(config:Dict[str,Any]) -> None:
    """
    Create parent directories for any file handlers that specify a filename.
    This prevents runtime errors when RotatingFileHandler points to logs/app.log.
    """
    handlers = config.get("handlers", {}) # Handlers means log processors. 
    if not isinstance(handlers,dict):
        return
    for h in handlers.values():
        if isinstance(h,dict) and "filename" in h:
            filename = Path(str(h["filename"]))
            filename.parent.mkdir(parents=True, exist_ok=True)