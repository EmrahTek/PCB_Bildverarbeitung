from __future__ import annotations

import logging
from pathlib import Path

import cv2 as cv
import numpy as np

from src.app.cli import parse_args
from src.app.pipeline import Pipeline, PipelineConfig
from src.camera_input.base import FrameSource
from src.camera_input.image import ImageFileConfig, ImageFileSource, ImageFolderConfig, ImageFolderSource
from src.camera_input.video_file import VideoFileConfig, VideoFileSource
from src.camera_input.webcam import WebcamConfig, WebcamSource
from src.detection_logic.board_first import BoardFirstConfig, BoardFirstDetector, ComponentSpec, RelativeROI
from src.detection_logic.template_match import TemplateMatchConfig, TemplateMatcher
from src.logging.setup import setup_logging
from src.preprocessing.geometry import BoardLocalizer, BoardWarpConfig
from src.utils.io import first_existing_directory, load_bgr, load_templates, load_yaml, sample_evenly

LOGGER = logging.getLogger(__name__)


class ResizePreprocessor:
    """Resize frames to a fixed width while keeping aspect ratio."""

    def __init__(self, width: int) -> None:
        if width <= 0:
            raise ValueError("width must be positive")
        self._width = width

    def process(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self._width:
            return frame
        scale = self._width / float(width)
        return cv.resize(frame, (self._width, int(round(height * scale))), interpolation=cv.INTER_AREA)

def build_source(args) -> FrameSource:
    """Construct the requested frame source from CLI arguments."""
    if args.source == "webcam":
        return WebcamSource(
            WebcamConfig(
                index=args.camera_index,
                width=args.width,
                height=args.height,
                target_fps=args.camera_fps,
            )
        )

    if args.source == "video":
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

    if args.source == "image":
        if args.image_path is None:
            raise ValueError("--image-path is required when --source image")
        return ImageFileSource(ImageFileConfig(path=args.image_path, loop=args.loop))

    if args.source == "images":
        if args.images_dir is None:
            raise ValueError("--images-dir is required when --source images")
        return ImageFolderSource(ImageFolderConfig(directory=args.images_dir, loop=args.loop, recursive=args.recursive))

    raise ValueError(f"Unsupported source: {args.source}")


def _tuple_floats(values: list[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _component_specs_and_matchers(config: dict) -> tuple[list[ComponentSpec], dict[str, TemplateMatcher]]:
    components_cfg = config["components"]
    template_dirs_cfg = config["templates"]["component_dirs"]
    specs: list[ComponentSpec] = []
    matchers: dict[str, TemplateMatcher] = {}

    for label, component_cfg in components_cfg.items():
        candidates = template_dirs_cfg.get(label, [])
        template_dir = first_existing_directory(candidates)
        if template_dir is None:
            LOGGER.warning("Skipping %s because no template directory was found in %s", label, candidates)
            continue

        templates = load_templates(template_dir)
        if not templates:
            LOGGER.warning("Skipping %s because the template directory is empty: %s", label, template_dir)
            continue

        # The canonical board view removes most rotation variance, so we only need a
        # compact scale bank here. That keeps webcam performance acceptable.
        templates = sample_evenly(templates, min(len(templates), 12))
        matcher = TemplateMatcher(
            templates,
            TemplateMatchConfig(
                label=label,
                score_threshold=float(component_cfg["score_threshold"]),
                scales=_tuple_floats(component_cfg["scales"]),
                gray_weight=float(component_cfg["gray_weight"]),
                edge_weight=float(component_cfg["edge_weight"]),
                use_clahe=bool(component_cfg["use_clahe"]),
                blur_ksize=int(component_cfg["blur_ksize"]),
                min_template_size=int(component_cfg["min_template_size"]),
            ),
        )
        roi = component_cfg["roi"]
        specs.append(
            ComponentSpec(
                label=label,
                roi=RelativeROI(float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3])),
                score_threshold=float(component_cfg["score_threshold"]),
            )
        )
        matchers[label] = matcher

    return specs, matchers


def build_detector(config: dict, source: str) -> BoardFirstDetector:
    """Create the full board-first detector from the YAML config."""
    board_cfg = config["board"]
    tracking_cfg = config["tracking"]
    templates_cfg = config["templates"]

    board_dir = first_existing_directory(templates_cfg["board_dirs"])
    reference_boards: list[np.ndarray] = []
    if board_dir is not None:
        board_paths = sample_evenly(
            [Path(path) for path in sorted(board_dir.glob("*")) if path.is_file()],
            int(board_cfg["max_reference_templates"]),
        )
        reference_boards = [load_bgr(path) for path in board_paths]
        LOGGER.info("Loaded %d board reference images from %s", len(reference_boards), board_dir)
    else:
        LOGGER.warning("No board reference directory found in %s", templates_cfg["board_dirs"])

    localizer = BoardLocalizer(
        cfg=BoardWarpConfig(
            output_size=(int(board_cfg["output_size"][0]), int(board_cfg["output_size"][1])),
            blur_ksize=int(board_cfg["blur_ksize"]),
            canny_t1=int(board_cfg["canny_t1"]),
            canny_t2=int(board_cfg["canny_t2"]),
            min_area_ratio=float(board_cfg["min_area_ratio"]),
            max_area_ratio=float(board_cfg["max_area_ratio"]),
            min_rectangularity=float(board_cfg["min_rectangularity"]),
            expected_aspect_ratio=float(board_cfg["expected_aspect_ratio"]),
            min_aspect_ratio=float(board_cfg["min_aspect_ratio"]),
            max_aspect_ratio=float(board_cfg["max_aspect_ratio"]),
            border_margin=int(board_cfg["border_margin"]),
            close_kernel=int(board_cfg["close_kernel"]),
            open_kernel=int(board_cfg["open_kernel"]),
            search_expansion=float(board_cfg["search_expansion"]),
            min_score=float(board_cfg["min_score"]),
            min_tracked_score=float(board_cfg["min_tracked_score"]),
            verify_gray_weight=float(board_cfg["verify_gray_weight"]),
            verify_edge_weight=float(board_cfg["verify_edge_weight"]),
        ),
        reference_boards=reference_boards,
    )

    specs, matchers = _component_specs_and_matchers(config)
    LOGGER.info("Enabled component matchers: %s", sorted(matchers.keys()))

    temporal_min_hits = int(tracking_cfg["temporal_min_hits"]) if source in {"webcam", "video"} else 1

    return BoardFirstDetector(
        localizer=localizer,
        component_matchers=matchers,
        component_specs=specs,
        cfg=BoardFirstConfig(
            temporal_window=int(tracking_cfg["temporal_window"]),
            temporal_min_hits=temporal_min_hits,
            max_missing_frames=int(tracking_cfg["max_missing_frames"]),
        ),
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.logging)
    config = load_yaml(args.config)

    runtime_cfg = config.get("runtime", {})
    detector = build_detector(config, args.source)
    source = build_source(args)

    resize_width = args.proc_resize_width if args.proc_resize_width is not None else runtime_cfg.get("processing_width")
    preprocessor = ResizePreprocessor(int(resize_width)) if resize_width else None

    pipeline = Pipeline(
        detector,
        preprocessor=preprocessor,
        cfg=PipelineConfig(
            window_name=str(runtime_cfg.get("window_name", "PCB Component Detection")),
            exit_key=str(runtime_cfg.get("exit_key", "q")),
        ),
    )
    pipeline.run(
        source,
        debug=args.debug,
        headless=args.headless,
        max_frames=args.max_frames,
        wait_ms=args.wait_ms,
    )


if __name__ == "__main__":
    main()
