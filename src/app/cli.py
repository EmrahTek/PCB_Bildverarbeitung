"""Command-line argument parsing for the PCB detection runtime.

The parser centralizes camera, image, video, debugging, and processing options
so main.py can build the requested source and detector pipeline.

Python docs:
- argparse: https://docs.python.org/3/library/argparse.html
- pathlib: https://docs.python.org/3/library/pathlib.html
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse runtime CLI arguments."""
    parser = argparse.ArgumentParser(description="FireBeetle V4 PCB component detection")
    parser.add_argument("--source", choices=["webcam", "ids", "picamera", "video", "image", "images"], default="webcam")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--logging", type=Path, default=Path("config/logging.yaml"))

    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-device", type=str, default=None)
    parser.add_argument("--camera-backend", type=str, default="auto")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument("--camera-buffer", type=int, default=1)
    parser.add_argument("--disable-mjpg", action="store_true")
    parser.add_argument(
        "--ids-allow-unverified-opencv",
        action="store_true",
        help="Compatibility flag; manual IDS OpenCV targets are allowed with a warning by default.",
    )
    parser.add_argument(
        "--list-video-devices",
        action="store_true",
        help="List /dev/video* devices with Linux sysfs names and exit.",
    )
    parser.add_argument(
        "--save-first-frame",
        type=Path,
        default=None,
        help="With --camera-open-check, save the first captured frame to this image path.",
    )
    parser.add_argument(
        "--camera-open-check",
        action="store_true",
        help="Open the selected live camera, read one frame, log diagnostics, and exit.",
    )

    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--loop", action="store_true")

    parser.add_argument("--video-resize-width", type=int, default=None)
    parser.add_argument("--video-resize-height", type=int, default=None)
    parser.add_argument("--video-stride", type=int, default=1)

    parser.add_argument("--proc-resize-width", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--wait-ms", type=int, default=1)
    return parser.parse_args(argv)
