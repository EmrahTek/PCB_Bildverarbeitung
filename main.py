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
    def __init__(self, width: int) -> None:
        self._width = int(width)

    def process(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self._width:
            return frame
        scale = self._width / w
        return cv.resize(frame, (self._width, int(round(h * scale))), interpolation=cv.INTER_AREA)


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
    if len(templates) <= max_count:
        return templates
    idxs = np.linspace(0, len(templates) - 1, num=max_count, dtype=int)
    return [templates[i] for i in idxs]


def augment_rotations(templates: list[np.ndarray], *, rotations: tuple[int, ...]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for t in templates:
        out.append(t)
        if 90 in rotations:
            out.append(cv.rotate(t, cv.ROTATE_90_CLOCKWISE))
        if 180 in rotations:
            out.append(cv.rotate(t, cv.ROTATE_180))
        if 270 in rotations:
            out.append(cv.rotate(t, cv.ROTATE_90_COUNTERCLOCKWISE))
    return out


def make_esp32_in_board_config(source: str) -> TemplateMatchConfig:
    if source.lower() in ("webcam", "video"):
        return TemplateMatchConfig(
            label="ESP32",
            score_threshold=0.60,
            scales=(0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30),
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
        score_threshold=0.58,
        scales=(0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35),
        nms_iou_threshold=0.22,
        max_candidates_per_template=8,
        max_detections=8,
        top_k=1,
        use_clahe=True,
        blur_ksize=3,
        local_max_kernel=5,
    )


def make_component_in_board_config(label: str, source: str) -> TemplateMatchConfig:
    src = source.lower()

    if label == "USB_PORT":
        score = 0.54 if src in ("image", "images") else 0.58
        scales = (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30)
    elif label == "JST_CONNECTOR":
        score = 0.50 if src in ("image", "images") else 0.54
        scales = (0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25)
    else:
        score = 0.52 if src in ("image", "images") else 0.56
        scales = (0.70, 0.80, 0.90, 1.00, 1.10, 1.20)

    return TemplateMatchConfig(
        label=label,
        score_threshold=score,
        scales=scales,
        nms_iou_threshold=0.20,
        max_candidates_per_template=6,
        max_detections=4,
        top_k=1,
        use_clahe=True,
        blur_ksize=3,
        local_max_kernel=5,
    )


def make_direct_fallback_config(source: str) -> TemplateMatchConfig:
    src = source.lower()

    if src in ("webcam", "video"):
        return TemplateMatchConfig(
            label="ESP32",
            score_threshold=0.70,
            scales=(0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.36, 0.42),
            nms_iou_threshold=0.18,
            max_candidates_per_template=6,
            max_detections=6,
            top_k=1,
            use_clahe=True,
            blur_ksize=3,
            local_max_kernel=5,
        )

    return TemplateMatchConfig(
        label="ESP32",
        score_threshold=0.68,
        scales=(0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.36, 0.42, 0.50),
        nms_iou_threshold=0.18,
        max_candidates_per_template=8,
        max_detections=8,
        top_k=1,
        use_clahe=True,
        blur_ksize=3,
        local_max_kernel=5,
    )


def make_direct_board_config(source: str) -> TemplateMatchConfig:
    src = source.lower()

    if src in ("webcam", "video"):
        return TemplateMatchConfig(
            label="BOARD",
            score_threshold=0.58,
            scales=(0.12, 0.16, 0.20, 0.24, 0.30, 0.36, 0.44, 0.52),
            nms_iou_threshold=0.20,
            max_candidates_per_template=6,
            max_detections=4,
            top_k=1,
            use_clahe=True,
            blur_ksize=3,
            local_max_kernel=7,
        )

    return TemplateMatchConfig(
        label="BOARD",
        score_threshold=0.56,
        scales=(0.12, 0.16, 0.20, 0.24, 0.30, 0.36, 0.44, 0.52, 0.62),
        nms_iou_threshold=0.20,
        max_candidates_per_template=8,
        max_detections=4,
        top_k=1,
        use_clahe=True,
        blur_ksize=3,
        local_max_kernel=7,
    )


def _build_optional_component_matcher(template_dir: Path, label: str, source: str, max_count: int) -> TemplateMatcher | None:
    if not template_dir.exists() or not template_dir.is_dir():
        return None

    templates = load_templates(template_dir)
    if not templates:
        return None

    base = sample_templates_evenly(templates, max_count=max_count)
    augmented = augment_rotations(base, rotations=(180,))
    return TemplateMatcher(augmented, make_component_in_board_config(label, source))


def build_detector(args) -> BoardFirstEsp32Detector:
    source = args.source.lower()

    esp_dir = Path("assets/templates/esp32_module")
    board_dir = Path("assets/templates/board")
    usb_dir = Path("assets/templates/usb_port")
    jst_dir = Path("assets/templates/jst_connector")
    reset_dir = Path("assets/templates/reset_button")

    esp_templates = load_templates(esp_dir)
    if not esp_templates:
        raise RuntimeError(f"No ESP32 templates found in: {esp_dir}")

    board_templates = load_templates(board_dir) if board_dir.exists() and board_dir.is_dir() else []

    base_count = 4 if source in ("webcam", "video") else 6

    esp_base = sample_templates_evenly(esp_templates, max_count=base_count)
    esp_augmented = augment_rotations(esp_base, rotations=(90, 180, 270))

    esp32_in_board_matcher = TemplateMatcher(
        esp_augmented,
        make_esp32_in_board_config(source),
    )

    direct_fallback_matcher = TemplateMatcher(
        esp_base,
        make_direct_fallback_config(source),
    )

    direct_board_matcher = None
    board_base: list[np.ndarray] = []
    if board_templates:
        board_base = sample_templates_evenly(board_templates, max_count=base_count)
        direct_board_matcher = TemplateMatcher(
            augment_rotations(board_base, rotations=(180,)),
            make_direct_board_config(source),
        )

    usb_matcher = _build_optional_component_matcher(usb_dir, "USB_PORT", source, max_count=4)
    jst_matcher = _build_optional_component_matcher(jst_dir, "JST_CONNECTOR", source, max_count=4)
    reset_matcher = _build_optional_component_matcher(reset_dir, "RESET_BUTTON", source, max_count=4)

    board_cfg = BoardWarpConfig(
        output_size=(900, 460),
        expected_aspect_ratio=900 / 460,
        min_area_ratio=0.01,
        max_area_ratio=0.30,
        min_aspect_ratio=1.45,
        max_aspect_ratio=2.55,
        min_rectangularity=0.60,
        border_margin=8,
    )

    detector = BoardFirstEsp32Detector(
        esp32_in_board_matcher=esp32_in_board_matcher,
        cfg=BoardFirstEsp32Config(
            board_cfg=board_cfg,
            fallback_direct_esp32=True,
            esp32_min_score_after_warp=0.58 if source in ("image", "images") else 0.60,
            usb_min_score_after_warp=0.54 if source in ("image", "images") else 0.58,
            jst_min_score_after_warp=0.52 if source in ("image", "images") else 0.56,
            reset_min_score_after_warp=0.54 if source in ("image", "images") else 0.58,
            direct_board_min_score=0.56 if source in ("image", "images") else 0.58,
            direct_esp32_min_score=0.68 if source in ("image", "images") else 0.70,
        ),
        direct_fallback_matcher=direct_fallback_matcher,
        direct_board_matcher=direct_board_matcher,
        usb_in_board_matcher=usb_matcher,
        jst_in_board_matcher=jst_matcher,
        reset_in_board_matcher=reset_matcher,
    )

    LOGGER.info(
        "Cascade detector active: geometry board -> direct board template -> direct ESP32 fallback -> ROI components."
    )
    LOGGER.info(
        "Loaded ESP32 templates from %s (raw=%d, used_base=%d, used_augmented=%d, source=%s)",
        esp_dir,
        len(esp_templates),
        len(esp_base),
        len(esp_augmented),
        source,
    )
    LOGGER.info(
        "Loaded BOARD templates from %s (raw=%d, used_base=%d, source=%s)",
        board_dir,
        len(board_templates),
        len(board_base),
        source,
    )
    LOGGER.info(
        "Optional component templates: usb=%s jst=%s reset=%s",
        usb_matcher is not None,
        jst_matcher is not None,
        reset_matcher is not None,
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