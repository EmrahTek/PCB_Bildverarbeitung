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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Args:
        argv: Optional argument list for testing. If None, uses sys.argv.

    Returns:
        argparse.Namespace with parsed runtime parameters.
    """
    p = argparse.ArgumentParser(description="PCB Component Detection (Real-time)")

    # Keep source values lowercase everywhere to avoid mismatches.
    p.add_argument(
        "--source",
        choices=["webcam", "video", "image"],
        default="webcam",
        help="Frame source type",
    )

    # Webcam options
    p.add_argument("--camera-index", type=int, default=0, help="Webcam device index")
    p.add_argument("--width", type=int, default=None, help="Requested capture width (best-effort)")
    p.add_argument("--height", type=int, default=None, help="Requested capture height (best-effort)")

    # Video / image options
    p.add_argument("--video-path", type=Path, default=None, help="Path to input video file")
    p.add_argument("--image-path", type=Path, default=None, help="Path to input image file")
    p.add_argument("--loop", action="store_true", help="Loop video/image playback")

    # Runtime flags
    p.add_argument("--debug", action="store_true", help="Enable debug overlay info")
    p.add_argument("--headless", action="store_true", help="Run without GUI window (future use)")
    p.add_argument("--logging", type=Path, default=Path("config/logging.yaml"), help="Logging YAML path")

    return p.parse_args(argv)