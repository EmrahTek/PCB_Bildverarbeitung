# main.py
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.app.cli import parse_args
from src.app.pipeline import Pipeline
from src.logging.setup import setup_logging

from src.camera_input.webcam import WebcamSource, WebcamConfig
from src.camera_input.video_file import VideoFileSource, VideoFileConfig
from src.camera_input.image import ImageFileSource, ImageFileConfig, ImageFolderSource, ImageFolderConfig

from src.utils.io import load_templates
from src.detection_logic.template_match import TemplateMatcher, TemplateMatchConfig

from src.preprocessing.geometry import warp_board, BoardWarpConfig

LOGGER = logging.getLogger(__name__)


class BoardWarpPreprocessor:
    """
    Optional preprocessor:
    - tries to detect the PCB board and warp to canonical view
    - if detection fails, returns the original frame (no crash)
    """
    def __init__(self, cfg: BoardWarpConfig) -> None:
        self._cfg = cfg

    def process(self, frame: np.ndarray) -> np.ndarray:
        warped, _H = warp_board(frame, self._cfg)
        return warped if warped is not None else frame


def build_source(args):
    src = args.source.lower()

    if src == "webcam":
        return WebcamSource(WebcamConfig(index=args.camera_index, width=args.width, height=args.height))

    if src == "video":
        if args.video_path is None:
            raise ValueError("--video-path is required when --source video")
        return VideoFileSource(VideoFileConfig(path=args.video_path, loop=args.loop))

    if src == "image":
        if args.image_path is None:
            raise ValueError("--image-path is required when --source image")
        return ImageFileSource(ImageFileConfig(path=args.image_path, loop=args.loop))

    if src == "images":
        if args.images_dir is None:
            raise ValueError("--images-dir is required when --source images")
        return ImageFolderSource(ImageFolderConfig(directory=args.images_dir, loop=args.loop, recursive=args.recursive))

    raise ValueError(f"Unsupported source: {args.source}")


def main() -> None:
    args = parse_args()
    setup_logging(args.logging)

    LOGGER.info("App starting with args=%s", vars(args))

    # --- Templates ---
    template_dir = Path("assets/templates/esp32_module")
    templates = load_templates(template_dir)

    detector = TemplateMatcher(
        templates,
        TemplateMatchConfig(
            label="ESP32",
            score_threshold=0.78,
            scales=(0.8, 0.9, 1.0, 1.1, 1.2),
            nms_iou_threshold=0.3,
        ),
    )

    # --- Optional preprocessing ---
    pre = None
    if args.warp_board:
        pre = BoardWarpPreprocessor(BoardWarpConfig(output_size=(800, 600)))

    source = build_source(args)
    pipeline = Pipeline(detector, preprocessor=pre)
    pipeline.run(source, debug=args.debug, headless=args.headless, max_frames=args.max_frames)

    LOGGER.info("App finished.")


if __name__ == "__main__":
    main()