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
from src.camera_input.image import (
    ImageFileSource,
    ImageFileConfig,
    ImageFolderSource,
    ImageFolderConfig,
)

from src.utils.io import load_templates
from src.detection_logic.template_match import TemplateMatcher, TemplateMatchConfig
from src.detection_logic.board_first_esp32 import (
    BoardFirstEsp32Config,
    BoardFirstEsp32Detector,
)
from src.preprocessing.geometry import BoardWarpConfig

LOGGER = logging.getLogger(__name__)


class ResizePreprocessor:
    """Resize frames before detection to reduce load and improve FPS."""

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
    """Apply multiple preprocessors in sequence."""

    def __init__(self, steps: list[object]) -> None:
        self._steps = steps

    def process(self, frame: np.ndarray) -> np.ndarray:
        for step in self._steps:
            frame = step.process(frame)
        return frame


def build_source(args):
    """Create the selected input source from CLI arguments."""
    src = args.source.lower()

    if src == "webcam":
        return WebcamSource(
            WebcamConfig(
                index=args.camera_index,
                width=args.width,
                height=args.height,
            )
        )

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
        return ImageFileSource(
            ImageFileConfig(
                path=args.image_path,
                loop=args.loop,
            )
        )

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


def sample_templates_evenly(templates: list[np.ndarray], max_count: int) -> list[np.ndarray]:
    """Select templates evenly across the list."""
    if len(templates) <= max_count:
        return templates
    idxs = np.linspace(0, len(templates) - 1, num=max_count, dtype=int)
    return [templates[i] for i in idxs]


def augment_rotations(templates: list[np.ndarray], *, rotations: tuple[int, ...]) -> list[np.ndarray]:
    """Create rotated template variants."""
    out: list[np.ndarray] = []
    for template in templates:
        out.append(template)
        if 90 in rotations:
            out.append(cv.rotate(template, cv.ROTATE_90_CLOCKWISE))
        if 180 in rotations:
            out.append(cv.rotate(template, cv.ROTATE_180))
        if 270 in rotations:
            out.append(cv.rotate(template, cv.ROTATE_90_COUNTERCLOCKWISE))
    return out


def make_esp32_in_board_config(source: str) -> TemplateMatchConfig:
    """
    Template matching config for ESP32 detection inside the warped board.
    """
    src = source.lower()

    if src in ("webcam", "video"):
        return TemplateMatchConfig(
            label="ESP32",
            score_threshold=0.66,
            scales=(0.72, 0.82, 0.92, 1.00, 1.08, 1.18, 1.28),
            nms_iou_threshold=0.22,
            max_candidates_per_template=6,
            max_detections=6,
            top_k=1,
            use_clahe=True,
            blur_ksize=3,
            local_max_kernel=5,
        )

    return TemplateMatchConfig(
        label="ESP32",
        score_threshold=0.62,
        scales=(0.68, 0.78, 0.88, 0.98, 1.08, 1.18, 1.30),
        nms_iou_threshold=0.22,
        max_candidates_per_template=8,
        max_detections=8,
        top_k=1,
        use_clahe=True,
        blur_ksize=3,
        local_max_kernel=5,
    )


def make_direct_fallback_config(source: str) -> TemplateMatchConfig:
    """
    Direct full-frame fallback config for ESP32.
    """
    src = source.lower()

    if src in ("webcam", "video"):
        return TemplateMatchConfig(
            label="ESP32",
            score_threshold=0.80,
            scales=(0.18, 0.22, 0.26, 0.30),
            nms_iou_threshold=0.20,
            max_candidates_per_template=4,
            max_detections=4,
            top_k=1,
            use_clahe=True,
            blur_ksize=3,
            local_max_kernel=5,
        )

    return TemplateMatchConfig(
        label="ESP32",
        score_threshold=0.84,
        scales=(0.16, 0.20, 0.24, 0.28, 0.32, 0.40),
        nms_iou_threshold=0.20,
        max_candidates_per_template=4,
        max_detections=4,
        top_k=1,
        use_clahe=True,
        blur_ksize=3,
        local_max_kernel=5,
    )


def build_detector(args) -> BoardFirstEsp32Detector:
    """
    Stable detector build:
    use the legacy board-first detector as the main detector.

    Reason:
    the hybrid detector is currently too unstable and produces false
    board detections on window/wall regions. For this project stage,
    the geometry-first + in-board ESP32 matcher is more reliable.
    """
    source = args.source.lower()

    esp_dir = Path("assets/templates/esp32_module")
    esp_templates = load_templates(esp_dir)
    if not esp_templates:
        raise RuntimeError(f"No ESP32 templates found in: {esp_dir}")

    if source in ("webcam", "video"):
        base_templates = sample_templates_evenly(esp_templates, max_count=3)
    else:
        base_templates = sample_templates_evenly(esp_templates, max_count=4)

    esp_augmented = augment_rotations(base_templates, rotations=(90, 180, 270))

    esp32_in_board_matcher = TemplateMatcher(
        esp_augmented,
        make_esp32_in_board_config(source),
    )

    direct_fallback_matcher = TemplateMatcher(
        base_templates,
        make_direct_fallback_config(source),
    )

    board_cfg = BoardWarpConfig(
        output_size=(900, 460),
        expected_aspect_ratio=900 / 460,
        min_area_ratio=0.01,
        max_area_ratio=0.30,
        min_rectangularity=0.60,
        border_margin=8,
    )

    detector = BoardFirstEsp32Detector(
        esp32_in_board_matcher=esp32_in_board_matcher,
        cfg=BoardFirstEsp32Config(
            board_cfg=board_cfg,
            fallback_direct_esp32=(source in ("webcam", "video")),
            esp32_min_score_after_warp=0.62 if source in ("image", "images") else 0.66,
        ),
        direct_fallback_matcher=direct_fallback_matcher,
    )

    LOGGER.info(
        "Legacy board-first detector active: geometry/homography for BOARD, "
        "template matching for ESP32 inside warped board."
    )
    LOGGER.info(
        "Loaded ESP32 templates from %s (raw=%d, used_base=%d, used_augmented=%d, source=%s)",
        esp_dir,
        len(esp_templates),
        len(base_templates),
        len(esp_augmented),
        source,
    )

    return detector


def main() -> None:
    args = parse_args()
    setup_logging(args.logging)

    LOGGER.info("App starting with args=%s", vars(args))

    steps: list[object] = []
    if args.proc_resize_width is not None:
        steps.append(ResizePreprocessor(args.proc_resize_width))
    preprocessor = ComposePreprocessor(steps) if steps else None

    detector = build_detector(args)
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