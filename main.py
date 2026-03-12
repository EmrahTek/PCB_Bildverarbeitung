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
from src.detection_logic.template_match import TemplateMatcher, TemplateMatchConfig
from src.logging.setup import setup_logging
from src.preprocessing.geometry import BoardWarpConfig, warp_board
from src.utils.io import load_templates

LOGGER = logging.getLogger(__name__)


class BoardWarpPreprocessor:
    """Detect board and warp it to a portrait canonical view."""

    def __init__(self, cfg: BoardWarpConfig) -> None:
        self._cfg = cfg

    def process(self, frame: np.ndarray) -> np.ndarray:
        warped, _H = warp_board(frame, self._cfg)
        return warped if warped is not None else frame


class ResizePreprocessor:
    """Resize frames before detection to reduce compute cost."""

    def __init__(self, width: int) -> None:
        self._width = int(width)

    def process(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self._width:
            return frame
        scale = self._width / w
        return cv.resize(frame, (self._width, int(round(h * scale))), interpolation=cv.INTER_AREA)


class ComposePreprocessor:
    """Chain multiple preprocessors in order."""

    def __init__(self, steps: list[object]) -> None:
        self._steps = steps

    def process(self, frame: np.ndarray) -> np.ndarray:
        for step in self._steps:
            frame = step.process(frame)
        return frame


def make_esp32_config(source: str, *, warp_enabled: bool) -> TemplateMatchConfig:
    src = source.lower()

    # Without board warp we still need small scales because the board may occupy
    # only part of the full image.
    if not warp_enabled:
        if src in ("image", "images"):
            return TemplateMatchConfig(
                label="ESP32",
                score_threshold=0.73,
                scales=(0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75),
                nms_iou_threshold=0.25,
                top_k=1,
                use_edges=True,
                edge_weight=0.35,
            )

        return TemplateMatchConfig(
            label="ESP32",
            score_threshold=0.78,
            scales=(0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.50),
            nms_iou_threshold=0.20,
            top_k=1,
            use_edges=True,
            edge_weight=0.30,
        )

    # With board warp enabled the search can focus on template scales near the
    # canonical board size. This is both more accurate and faster.
    if src in ("image", "images"):
        return TemplateMatchConfig(
            label="ESP32",
            score_threshold=0.66,
            scales=(0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25),
            nms_iou_threshold=0.25,
            top_k=1,
            use_edges=True,
            edge_weight=0.35,
        )

    return TemplateMatchConfig(
        label="ESP32",
        score_threshold=0.70,
        scales=(0.75, 0.85, 0.95, 1.05, 1.15),
        nms_iou_threshold=0.20,
        top_k=1,
        use_edges=True,
        edge_weight=0.35,
    )


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
        return ImageFolderSource(ImageFolderConfig(directory=args.images_dir, loop=args.loop, recursive=args.recursive))

    raise ValueError(f"Unsupported source: {args.source}")


def _augment_templates_for_source(templates: list[np.ndarray], source: str) -> list[np.ndarray]:
    src = source.lower()
    if src not in ("image", "images"):
        return templates

    # For offline image tuning we want strong rotation tolerance.
    augmented: list[np.ndarray] = []
    for t in templates:
        augmented.append(t)
        augmented.append(cv.rotate(t, cv.ROTATE_180))
        augmented.append(cv.rotate(t, cv.ROTATE_90_CLOCKWISE))
        augmented.append(cv.rotate(t, cv.ROTATE_90_COUNTERCLOCKWISE))
    return augmented


def main() -> None:
    args = parse_args()
    setup_logging(args.logging)
    LOGGER.info("App starting with args=%s", vars(args))

    template_dir = Path("assets/templates/esp32_module")
    templates = load_templates(template_dir)
    templates = _augment_templates_for_source(templates, args.source)

    detector = TemplateMatcher(
        templates,
        make_esp32_config(args.source, warp_enabled=bool(args.warp_board)),
    )

    steps: list[object] = []
    if args.proc_resize_width is not None:
        steps.append(ResizePreprocessor(args.proc_resize_width))

    if args.warp_board:
        steps.append(
            BoardWarpPreprocessor(
                BoardWarpConfig(
                    output_size=(480, 960),
                    canny_t1=40,
                    canny_t2=140,
                    min_area_ratio=0.08,
                    approx_eps_ratio=0.02,
                    blur_ksize=5,
                    morph_kernel=5,
                )
            )
        )

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
