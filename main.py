# main.py
from __future__ import annotations

import logging

import numpy as np

from src.app.cli import parse_args
from src.app.pipeline import Pipeline
from src.logging.setup import setup_logging

from src.camera_input.webcam import WebcamSource, WebcamConfig
from src.camera_input.video_file import VideoFileSource, VideoFileConfig

LOGGER = logging.getLogger(__name__)


class NoOpDetector:
    """
    Placeholder detector used until real detection logic is implemented.

    It always returns an empty detection list, allowing us to validate:
    - camera input
    - pipeline loop
    - FPS overlay
    - logging
    """
    def detect(self, frame: np.ndarray):
        return []


def build_source(args):
    """Factory for FrameSource based on CLI args."""
    src = args.source.lower()  # defensive normalization

    if src == "webcam":
        cfg = WebcamConfig(index=args.camera_index, width=args.width, height=args.height)
        return WebcamSource(cfg)

    if src == "video":
        if args.video_path is None:
            raise ValueError("--video-path is required when --source video")
        return VideoFileSource(VideoFileConfig(path=args.video_path, loop=args.loop))

    raise ValueError(f"Unsupported source: {args.source}")


def main() -> None:
    args = parse_args()
    setup_logging(args.logging)

    LOGGER.info("App starting with args=%s", vars(args))

    # Use a placeholder detector for Milestone-1.
    detector = NoOpDetector()

    source = build_source(args)
    pipeline = Pipeline(detector)
    pipeline.run(source, debug=args.debug, headless=args.headless)

    LOGGER.info("App finished.")


if __name__ == "__main__":
    main()