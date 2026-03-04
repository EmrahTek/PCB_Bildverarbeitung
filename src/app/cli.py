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

from __future__ import annotations

import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace: 
    """
    Parse CLI arguments.
    Keep interface stable early; add more options later.
    """
    p = argparse.ArgumentParser(description="PCB Component Detection (Real-time)")
    p.add_argument("--source", choices=["Webcam", "Video", "Foto"], default="webcam", help="Frame source type")
    p.add_argument("--camera-index", type=int, default=0, help="Webcam device index")
    p.add_argument("--width", type=int, default=None, help="Requested capture width (best-effort)")
    p.add_argument("--height", type=int, default=None, help="Requested capture height (best-effort)")
    p.add_argument("--video-path", type=Path, default=None, help="Path to input video file")
    p.add_argument("--loop", action="store_true", help="Loop video file playback")
    p.add_argument("--debug", action="store_true", help="Enable debug overlay info")
    p.add_argument("--headless", action="store_true", help="Run without GUI window (future use)")
    p.add_argument("--logging", type=Path, default=Path("config/logging.yaml"), help="Logging YAML path")
    return p.parse_args()

