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

    # Video / image / images options
    p.add_argument("--video-path", type=Path, default=None, help="Path to input video file")
    p.add_argument("--image-path", type=Path, default=None, help="Path to input image file")
    p.add_argument("--images-dir", type=Path, default=None, help="Directory containing images")
    p.add_argument("--recursive", action="store_true", help="Search images-dir recursively")
    p.add_argument("--loop", action="store_true", help="Loop playback for video/image/images")

    # Video performance options
    p.add_argument("--video-resize-width", type=int, default=None,
                help="Resize video frames by width (keeps aspect ratio if height is not set)")
    p.add_argument("--video-resize-height", type=int, default=None,
                help="Resize video frames by height (keeps aspect ratio if width is not set)")
    p.add_argument("--video-stride", type=int, default=1,
                help="Read every Nth frame (stride>=1). 2=skip 1 decode 1, ...")

    # runtime
    p.add_argument("--debug", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--logging", type=Path, default=Path("config/logging.yaml"))

    # preprocessing flags (optional)
    p.add_argument("--warp-board", action="store_true", help="Enable board detection + homography warp")

    p.add_argument("--wait-ms", type=int, default=1, help="Delay per frame in ms (images: try 3000)")
    p.add_argument("--proc-resize-width", type=int, default=None, help="Resize input frames for processing (keeps aspect ratio)")

    return p.parse_args(argv)