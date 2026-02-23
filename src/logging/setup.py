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