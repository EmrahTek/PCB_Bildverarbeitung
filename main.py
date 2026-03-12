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
from src.detection_logic.board_detector import BoardDetectorConfig, BoardEsp32Detector, Esp32InBoardConfig
from src.logging.setup import setup_logging
from src.utils.io import load_templates

LOGGER = logging.getLogger(__name__)


class ResizePreprocessor:
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


def _find_existing_dir(candidates: list[str]) -> Path:
    for c in candidates:
        p = Path(c)
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError("No template directory found from candidates: " + ", ".join(candidates))


def _sample_templates_for_live(templates: list[np.ndarray], keep: int = 4) -> list[np.ndarray]:
    if len(templates) <= keep:
        return templates
    idx = np.linspace(0, len(templates) - 1, keep, dtype=int)
    return [templates[i] for i in idx]


def _augment_templates(templates: list[np.ndarray], *, source: str, full_rot: bool) -> list[np.ndarray]:
    src = source.lower()
    if src in ("webcam", "video"):
        rotations = (
            lambda x: x,
            lambda x: cv.rotate(x, cv.ROTATE_180),
        )
    else:
        rotations = (
            lambda x: x,
            lambda x: cv.rotate(x, cv.ROTATE_180),
        ) if not full_rot else (
            lambda x: x,
            lambda x: cv.rotate(x, cv.ROTATE_90_CLOCKWISE),
            lambda x: cv.rotate(x, cv.ROTATE_180),
            lambda x: cv.rotate(x, cv.ROTATE_90_COUNTERCLOCKWISE),
        )

    out: list[np.ndarray] = []
    for t in templates:
        for fn in rotations:
            out.append(fn(t))
    return out


def make_board_config(source: str, matcher_profile: str = "balanced") -> BoardDetectorConfig:
    src = source.lower()
    profile = matcher_profile.lower()

    if src in ("webcam", "video"):
        if profile == "fast":
            scales = (0.12, 0.16, 0.20, 0.26, 0.34, 0.44)
            threshold = 0.58
        elif profile == "accurate":
            scales = (0.10, 0.12, 0.16, 0.20, 0.26, 0.34, 0.44, 0.56)
            threshold = 0.56
        else:
            scales = (0.10, 0.12, 0.16, 0.20, 0.26, 0.34, 0.44, 0.56)
            threshold = 0.57
        return BoardDetectorConfig(
            label="BOARD",
            score_threshold=threshold,
            scales=scales,
            use_edges=True,
            edge_weight=0.40,
            allow_tracking=True,
            search_margin_px=110,
            roi_expand_ratio=0.25,
            tracking_max_misses=14,
        )

    if profile == "fast":
        scales = (0.12, 0.16, 0.20, 0.26, 0.34, 0.44, 0.56, 0.72)
        threshold = 0.60
    elif profile == "accurate":
        scales = (0.10, 0.12, 0.14, 0.16, 0.20, 0.26, 0.34, 0.44, 0.56, 0.72, 0.90)
        threshold = 0.57
    else:
        scales = (0.10, 0.12, 0.14, 0.16, 0.20, 0.26, 0.34, 0.44, 0.56, 0.72, 0.90)
        threshold = 0.58
    return BoardDetectorConfig(
        label="BOARD",
        score_threshold=threshold,
        scales=scales,
        use_edges=True,
        edge_weight=0.36,
        allow_tracking=False,
        search_margin_px=90,
        roi_expand_ratio=0.20,
    )


def make_esp32_in_board_config(source: str, matcher_profile: str = "balanced") -> Esp32InBoardConfig:
    src = source.lower()
    profile = matcher_profile.lower()

    if src in ("webcam", "video"):
        if profile == "fast":
            scales = (0.50, 0.65, 0.80, 0.95, 1.10)
            threshold = 0.66
        elif profile == "accurate":
            scales = (0.45, 0.55, 0.65, 0.80, 0.95, 1.10, 1.25)
            threshold = 0.62
        else:
            scales = (0.45, 0.55, 0.65, 0.80, 0.95, 1.10, 1.25)
            threshold = 0.64
    else:
        if profile == "fast":
            scales = (0.50, 0.65, 0.80, 0.95, 1.10, 1.25)
            threshold = 0.70
        elif profile == "accurate":
            scales = (0.40, 0.50, 0.60, 0.75, 0.90, 1.05, 1.20, 1.35)
            threshold = 0.66
        else:
            scales = (0.40, 0.50, 0.60, 0.75, 0.90, 1.05, 1.20, 1.35)
            threshold = 0.68

    return Esp32InBoardConfig(
        label="ESP32",
        score_threshold=threshold,
        scales=scales,
        use_edges=True,
        edge_weight=0.22,
        board_margin_ratio=0.03,
        search_rel_x=0.34,
        search_rel_y=0.40,
        search_rel_w=0.60,
        search_rel_h=0.58,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.logging)
    LOGGER.info("App starting with args=%s", vars(args))

    source_name = args.source.lower()
    matcher_profile = (getattr(args, "matcher_profile", "balanced") or "balanced").lower()

    board_template_dir = _find_existing_dir([
        "assets/templates/BoardFireBeetle",
        "assets/templates/board_firebeetle",
        "assets/templates/board",
    ])
    esp32_template_dir = _find_existing_dir([
        "assets/templates/esp32_module",
        "assets/templates/ESP32",
    ])

    board_templates = load_templates(board_template_dir)
    esp32_templates = load_templates(esp32_template_dir)

    live_mode = source_name in ("webcam", "video")
    if live_mode:
        board_templates = _sample_templates_for_live(board_templates, keep=6)
        esp32_templates = _sample_templates_for_live(esp32_templates, keep=4)

    board_templates = _augment_templates(board_templates, source=source_name, full_rot=not live_mode)
    esp32_templates = _augment_templates(esp32_templates, source=source_name, full_rot=not live_mode)

    board_cfg = make_board_config(source_name, matcher_profile=matcher_profile)
    esp32_cfg = make_esp32_in_board_config(source_name, matcher_profile=matcher_profile)
    detector = BoardEsp32Detector(board_templates, board_cfg, esp32_templates, esp32_cfg)

    LOGGER.info(
        "Loaded BOARD detector from %s (templates=%d, source=%s, profile=%s)",
        board_template_dir,
        len(board_templates),
        source_name,
        matcher_profile,
    )
    LOGGER.info(
        "Loaded ESP32-in-board detector from %s (templates=%d, source=%s, profile=%s)",
        esp32_template_dir,
        len(esp32_templates),
        source_name,
        matcher_profile,
    )
    LOGGER.info("Board-first mode active: BOARD is detected first, ESP32 is searched only inside BOARD.")
    LOGGER.info("Qt font warnings from OpenCV/Qt can be ignored on Ubuntu; they do not affect detection.")

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
