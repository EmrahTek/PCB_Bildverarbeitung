from __future__ import annotations

import logging
from pathlib import Path

import cv2 as cv
import numpy as np

from src.app.cli import parse_args
from src.app.pipeline import Pipeline
from src.camera_input.image import ImageFileSource, ImageFileConfig, ImageFolderSource, ImageFolderConfig
from src.camera_input.video_file import VideoFileSource, VideoFileConfig
from src.camera_input.webcam import WebcamSource, WebcamConfig
from src.detection_logic.board_detector import BoardDetector, BoardDetectorConfig
from src.logging.setup import setup_logging
from src.utils.io import load_templates

LOGGER = logging.getLogger(__name__)


class ResizePreprocessor:
    """Resize frames before detection to reduce compute cost while keeping aspect ratio."""

    def __init__(self, width: int) -> None:
        self._width = int(width)

    def process(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self._width:
            return frame
        scale = self._width / w
        new_h = int(round(h * scale))
        return cv.resize(frame, (self._width, new_h), interpolation=cv.INTER_AREA)


class ComposePreprocessor:
    """Chain multiple preprocessors in order."""

    def __init__(self, steps: list[object]) -> None:
        self._steps = steps

    def process(self, frame: np.ndarray) -> np.ndarray:
        for step in self._steps:
            frame = step.process(frame)
        return frame


def build_source(args):
    src = args.source.lower()

    if src == "webcam":
        return WebcamSource(WebcamConfig(index=args.camera_index, width=args.width, height=args.height))

    if src == "video":
        if args.video_path is None:
            raise ValueError("--video-path is required when --source video")
        return VideoFileSource(
            VideoFileConfig(
                path=args.video_path,
                loop=args.loop,
                resize_width=args.video_resize_width,
                resize_height=args.video_resize_height,
                stride=args.video_stride,
            )
        )

    if src == "image":
        if args.image_path is None:
            raise ValueError("--image-path is required when --source image")
        return ImageFileSource(ImageFileConfig(path=args.image_path, loop=args.loop))

    if src == "images":
        if args.images_dir is None:
            raise ValueError("--images-dir is required when --source images")
        return ImageFolderSource(
            ImageFolderConfig(
                directory=args.images_dir,
                loop=args.loop,
                recursive=args.recursive,
            )
        )

    raise ValueError(f"Unsupported source: {args.source}")


def _find_board_template_dir() -> Path:
    candidates = [
        Path("assets/templates/BoardFireBeetle"),
        Path("assets/templates/board_firebeetle"),
        Path("assets/templates/board"),
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    raise FileNotFoundError(
        "Board template folder not found. Expected one of: "
        "assets/templates/BoardFireBeetle, assets/templates/board_firebeetle, assets/templates/board"
    )


def _sample_templates_for_live(templates: list[np.ndarray], keep: int = 6) -> list[np.ndarray]:
    if len(templates) <= keep:
        return templates
    idx = np.linspace(0, len(templates) - 1, keep, dtype=int)
    return [templates[i] for i in idx]


def _augment_board_templates(templates: list[np.ndarray], *, source: str) -> list[np.ndarray]:
    src = source.lower()
    out: list[np.ndarray] = []

    if src in ("image", "images"):
        rotations = (
            lambda x: x,
            lambda x: cv.rotate(x, cv.ROTATE_90_CLOCKWISE),
            lambda x: cv.rotate(x, cv.ROTATE_180),
            lambda x: cv.rotate(x, cv.ROTATE_90_COUNTERCLOCKWISE),
        )
    else:
        # live mode: keep augmentation lighter for speed
        rotations = (
            lambda x: x,
            lambda x: cv.rotate(x, cv.ROTATE_180),
        )

    for t in templates:
        for fn in rotations:
            out.append(fn(t))
    return out


def make_board_config(source: str, matcher_profile: str = "balanced") -> BoardDetectorConfig:
    src = source.lower()
    profile = matcher_profile.lower()

    if src in ("webcam", "video"):
        if profile == "fast":
            scales = (0.16, 0.20, 0.24, 0.30, 0.38, 0.48)
            threshold = 0.62
        elif profile == "accurate":
            scales = (0.14, 0.18, 0.22, 0.26, 0.30, 0.36, 0.44, 0.54)
            threshold = 0.60
        else:
            scales = (0.14, 0.18, 0.22, 0.26, 0.30, 0.36, 0.44, 0.54)
            threshold = 0.61
        return BoardDetectorConfig(
            label="BOARD",
            score_threshold=threshold,
            scales=scales,
            use_edges=True,
            edge_weight=0.38,
            allow_tracking=True,
            search_margin_px=96,
            roi_expand_ratio=0.22,
            tracking_max_misses=12,
        )

    # image / images tuning mode
    if profile == "fast":
        scales = (0.14, 0.18, 0.22, 0.26, 0.30, 0.36, 0.44, 0.54, 0.66)
        threshold = 0.63
    elif profile == "accurate":
        scales = (0.12, 0.14, 0.16, 0.18, 0.22, 0.26, 0.30, 0.36, 0.44, 0.54, 0.66, 0.80)
        threshold = 0.60
    else:
        scales = (0.12, 0.14, 0.16, 0.18, 0.22, 0.26, 0.30, 0.36, 0.44, 0.54, 0.66, 0.80)
        threshold = 0.61
    return BoardDetectorConfig(
        label="BOARD",
        score_threshold=threshold,
        scales=scales,
        use_edges=True,
        edge_weight=0.35,
        allow_tracking=False,
        search_margin_px=80,
        roi_expand_ratio=0.20,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.logging)
    LOGGER.info("App starting with args=%s", vars(args))

    board_template_dir = _find_board_template_dir()
    templates = load_templates(board_template_dir)

    if args.source.lower() in ("webcam", "video"):
        templates = _sample_templates_for_live(templates, keep=6)

    templates = _augment_board_templates(templates, source=args.source)

    matcher_profile = getattr(args, "matcher_profile", "balanced") or "balanced"
    detector_cfg = make_board_config(args.source, matcher_profile=matcher_profile)
    detector = BoardDetector(templates, detector_cfg)

    LOGGER.info(
        "Loaded BOARD detector from %s (templates=%d, source=%s, profile=%s)",
        board_template_dir,
        len(templates),
        args.source,
        matcher_profile,
    )
    LOGGER.info("Board-first mode active: components are disabled until BOARD is stable.")

    steps: list[object] = []
    if args.proc_resize_width is not None:
        steps.append(ResizePreprocessor(args.proc_resize_width))
    preprocessor = ComposePreprocessor(steps) if steps else None

    source = build_source(args)
    pipeline = Pipeline(detector, preprocessor=preprocessor)
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
