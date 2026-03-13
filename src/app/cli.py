# argparse / flags (camera vs video, headless, debug)
# Saubere CLI: Kamera vs Video, Debug, Headless, max-frames, config-path.
"""
cli.py (Command line interface)

This module defines the command line interface (CLI) for running the application.
It parses arguments such as:
- camera index or video file path
- headless mode
- debug mode
- config path
- max frames

Inputs:
- Command line arguments (argv)

Outputs:
- argparse.Namespace with validated runtime settings
- Loaded configuration dictionary



    Zu implementierende Funktionen

    build_arg_parser() -> argparse.ArgumentParser

    parse_args(argv=None) -> Namespace

    load_config(config_path: Path) -> dict

    (Optional) validate_args(args) -> None


   argparse:
    # https://docs.python.org/3/library/argparse.html     
"""

# src/app/cli.py
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the PCB detection application."""
    parser = argparse.ArgumentParser(description="PCB Component Detection")

    parser.add_argument("--source", choices=["webcam", "video", "image", "images"], default="webcam")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--logging", type=Path, default=Path("config/logging.yaml"))

    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=None)

    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--loop", action="store_true")

    parser.add_argument("--video-resize-width", type=int, default=None)
    parser.add_argument("--video-resize-height", type=int, default=None)
    parser.add_argument("--video-stride", type=int, default=1)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--wait-ms", type=int, default=1)
    parser.add_argument("--log-every-n", type=int, default=30)

    parser.add_argument("--proc-resize-width", type=int, default=None)
    parser.add_argument("--disable-board-warp", action="store_true")
    parser.add_argument("--matcher-profile", choices=["fast", "balanced", "accurate"], default="balanced")

    return parser.parse_args(argv)
