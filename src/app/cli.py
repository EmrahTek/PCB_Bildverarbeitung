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
    """
    CLI arguments for selecting source and runtime behavior.
    """
    p = argparse.ArgumentParser(description="PCB Component Detection (Real-time)")

    p.add_argument("--source", choices=["webcam", "video", "image", "images"], default="webcam")

    # webcam
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)

    # video / image / images
    p.add_argument("--video-path", type=Path, default=None)
    p.add_argument("--image-path", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--loop", action="store_true")

    # runtime
    p.add_argument("--debug", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--logging", type=Path, default=Path("config/logging.yaml"))

    # preprocessing flags (optional)
    p.add_argument("--warp-board", action="store_true", help="Enable board detection + homography warp")

    return p.parse_args(argv)