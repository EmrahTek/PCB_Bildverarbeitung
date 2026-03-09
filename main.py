# main.py
from __future__ import annotations

import logging
from pathlib import Path

import cv2 as cv
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
    """Detect board and warp to canonical view; fallback to original frame if not found."""
    def __init__(self, cfg: BoardWarpConfig) -> None:
        self._cfg = cfg

    def process(self, frame: np.ndarray) -> np.ndarray:
        warped, _H = warp_board(frame, self._cfg)
        return warped if warped is not None else frame


class ResizePreprocessor:
    """Resize frames before detection to avoid OOM and improve FPS."""
    def __init__(self, width: int) -> None:
        self._width = int(width)

    def process(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self._width:
            return frame
        scale = self._width / w
        new_w = self._width
        new_h = int(round(h * scale))
        return cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)


class ComposePreprocessor:
    """Chain multiple preprocessors in order."""
    def __init__(self, steps: list[object]) -> None:
        self._steps = steps

    def process(self, frame: np.ndarray) -> np.ndarray:
        for s in self._steps:
            frame = s.process(frame)
        return frame


def make_esp32_config(source: str) -> TemplateMatchConfig:
    """
    Choose detector parameters based on input source.

    - webcam/video: fast + strict to reduce false positives
    - image/images: tune profile for higher recall on dataset images
    """
    src = source.lower()

    if src in ("webcam", "video"):
        return TemplateMatchConfig(
            label="ESP32",
            score_threshold=0.88,
            scales=(0.18, 0.20, 0.22, 0.25, 0.28, 0.30),
            nms_iou_threshold=0.2,
            top_k=1,
        )

    return TemplateMatchConfig(
        label="ESP32",
        score_threshold=0.80,
        scales=(0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
        nms_iou_threshold=0.25,
        top_k=1,
    )


def build_source(args):
    src = args.source.lower()

    if src == "webcam":
        return WebcamSource(WebcamConfig(index=args.camera_index, width=args.width, height=args.height))

    if src == "video":
        if args.video_path is None:
            raise ValueError("--video-path is required when --source video")
        return VideoFileSource(VideoFileConfig(
            path=args.video_path,
            loop=args.loop,
            resize_width=args.video_resize_width,
            resize_height=args.video_resize_height,
            stride=args.video_stride,
        ))

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

    # Rotate templates only for image-based tuning (avoid FPS drop in webcam/video)
    if args.source.lower() in ("image", "images"):
        aug: list[np.ndarray] = []
        for t in templates:
            aug.append(t)
            aug.append(cv.rotate(t, cv.ROTATE_90_CLOCKWISE))
            aug.append(cv.rotate(t, cv.ROTATE_180))
            aug.append(cv.rotate(t, cv.ROTATE_90_COUNTERCLOCKWISE))
        templates = aug

    detector = TemplateMatcher(templates, make_esp32_config(args.source))

    # --- Optional preprocessing ---
    steps = []

    if args.proc_resize_width is not None:
        steps.append(ResizePreprocessor(args.proc_resize_width))

    if args.warp_board:
        steps.append(BoardWarpPreprocessor(BoardWarpConfig(output_size=(800, 600))))

    pre = ComposePreprocessor(steps) if steps else None

    source = build_source(args)
    pipeline = Pipeline(detector, preprocessor=pre)

    pipeline.run(
        source,
        debug=args.debug,
        headless=args.headless,
        max_frames=args.max_frames,
        wait_ms=args.wait_ms,
    )

    LOGGER.info("App finished.")


if __name__ == "__main__":
    main()