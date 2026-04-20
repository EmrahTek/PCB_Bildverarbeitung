from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from src.app.cli import parse_args
from src.app.pipeline import Pipeline, PipelineConfig
from src.camera_input.base import FrameSource
from src.camera_input.image import ImageFileConfig, ImageFileSource, ImageFolderConfig, ImageFolderSource
from src.camera_input.video_file import VideoFileConfig, VideoFileSource
from src.camera_input.webcam import WebcamConfig, WebcamSource
from src.detection_logic.board_first import BoardFirstConfig, BoardFirstDetector, ComponentSpec, RelativeROI
from src.detection_logic.coarse_board import BoardTemplateLocator, BoardTemplateLocatorConfig
from src.detection_logic.template_match import TemplateMatchConfig, TemplateMatcher
from src.logging.setup import setup_logging
from src.preprocessing.geometry import BoardLocalizer, BoardWarpConfig
from src.utils.io import first_existing_directory, load_bgr, load_templates, load_yaml, rotate_image, sample_evenly
from src.utils.prepared_templates import PreparedTemplateBank, load_prepared_template_bank

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


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge config overrides without mutating the input objects."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _apply_source_profile(config: dict[str, Any], source: str) -> dict[str, Any]:
    """Apply optional source-specific overrides from the YAML configuration."""
    profiles = config.get("source_profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("source_profiles must be a dictionary when present")
    override = profiles.get(source)
    if not isinstance(override, dict):
        return deepcopy(config)
    return _deep_merge_dicts(config, override)


def _existing_directories(candidates: list[str | Path]) -> list[Path]:
    return [Path(candidate).resolve() for candidate in candidates if Path(candidate).exists() and Path(candidate).is_dir()]


def _first_existing_file(candidates: list[str | Path]) -> Path | None:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def _load_prepared_template_bank(config: dict) -> PreparedTemplateBank | None:
    templates_cfg = config.get("templates", {})
    metadata_path = _first_existing_file(templates_cfg.get("prepared_metadata_paths", []))
    if metadata_path is None:
        return None
    rotation_turns = int(templates_cfg.get("rotation_turns", 0))
    return load_prepared_template_bank(metadata_path, rotation_turns=rotation_turns)


def _rotate_template_bank(images: list[np.ndarray], turns_90: int) -> list[np.ndarray]:
    if turns_90 % 4 == 0:
        return images
    return [rotate_image(image, turns_90) for image in images]


def _build_board_template_locator(config: dict, board_dirs: list[Path], template_rotation_turns: int) -> BoardTemplateLocator | None:
    board_template_cfg = config.get("board_template", {})
    if not bool(board_template_cfg.get("enabled", True)):
        return None
    if not board_dirs:
        return None

    max_templates = int(board_template_cfg.get("max_templates", 4))
    rotations = tuple(int(turn) for turn in board_template_cfg.get("rotations", [0, 1, 2, 3]))

    templates_gray: list[np.ndarray] = []
    all_templates: list[np.ndarray] = []
    for board_dir in board_dirs:
        all_templates.extend(_rotate_template_bank(load_templates(board_dir), template_rotation_turns))
    for template in sample_evenly(all_templates, min(len(all_templates), max_templates)):
        for turns_90 in rotations:
            templates_gray.append(rotate_image(template, turns_90))

    if not templates_gray:
        return None

    matcher = TemplateMatcher(
        templates_gray,
        TemplateMatchConfig(
            label="BOARD",
            score_threshold=float(board_template_cfg.get("score_threshold", 0.28)),
            scales=_tuple_floats(board_template_cfg.get("scales", [0.18, 0.22, 0.26, 0.30, 0.36, 0.42, 0.50, 0.60, 0.72, 0.86, 1.0])),
            gray_weight=float(board_template_cfg.get("gray_weight", 0.35)),
            edge_weight=float(board_template_cfg.get("edge_weight", 0.65)),
            use_clahe=bool(board_template_cfg.get("use_clahe", True)),
            blur_ksize=int(board_template_cfg.get("blur_ksize", 3)),
            min_template_size=int(board_template_cfg.get("min_template_size", 32)),
        ),
    )
    return BoardTemplateLocator(
        matcher,
        BoardTemplateLocatorConfig(
            resize_width=int(board_template_cfg.get("search_resize_width", config.get("runtime", {}).get("processing_width", 960))),
            min_score=float(board_template_cfg.get("score_threshold", 0.28)),
            max_candidates=int(board_template_cfg.get("max_candidates", 8)),
        ),
    )


def _component_specs_and_matchers(
    config: dict,
    prepared_bank: PreparedTemplateBank | None,
    template_rotation_turns: int,
) -> tuple[list[ComponentSpec], dict[str, TemplateMatcher]]:
    components_cfg = config["components"]
    template_dirs_cfg = config["templates"]["component_dirs"]
    use_prepared_rois = bool(config["templates"].get("use_prepared_rois", False))
    specs: list[ComponentSpec] = []
    matchers: dict[str, TemplateMatcher] = {}

    for label, component_cfg in components_cfg.items():
        candidates: list[str | Path] = []
        if prepared_bank is not None and label in prepared_bank.component_dirs:
            candidates.append(prepared_bank.component_dirs[label])
        candidates.extend(template_dirs_cfg.get(label, []))
        template_dir = first_existing_directory(candidates)
        if template_dir is None:
            LOGGER.warning("Skipping %s because no template directory was found in %s", label, candidates)
            continue

        templates = _rotate_template_bank(load_templates(template_dir), template_rotation_turns)
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
        roi = (
            prepared_bank.component_rois.get(label, tuple(component_cfg["roi"]))
            if prepared_bank is not None and use_prepared_rois
            else tuple(component_cfg["roi"])
        )
        layout_roi = None
        if prepared_bank is not None and label in prepared_bank.component_rois:
            layout_roi_values = prepared_bank.component_rois[label]
            layout_roi = RelativeROI(
                float(layout_roi_values[0]),
                float(layout_roi_values[1]),
                float(layout_roi_values[2]),
                float(layout_roi_values[3]),
            )
        elif "layout_roi" in component_cfg:
            layout_roi_values = tuple(component_cfg["layout_roi"])
            layout_roi = RelativeROI(
                float(layout_roi_values[0]),
                float(layout_roi_values[1]),
                float(layout_roi_values[2]),
                float(layout_roi_values[3]),
            )
        specs.append(
            ComponentSpec(
                label=label,
                roi=RelativeROI(float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3])),
                score_threshold=float(component_cfg["score_threshold"]),
                layout_roi=layout_roi,
                layout_fallback_score=float(component_cfg.get("layout_fallback_score", 0.0)),
                layout_fallback_min_board_score=float(component_cfg.get("layout_fallback_min_board_score", 0.60)),
                min_board_area_ratio=float(component_cfg.get("min_board_area_ratio", 0.0)),
                max_board_area_ratio=float(component_cfg.get("max_board_area_ratio", 1.0)),
                min_normalized_aspect_ratio=float(component_cfg.get("min_normalized_aspect_ratio", 1.0)),
                max_normalized_aspect_ratio=float(component_cfg.get("max_normalized_aspect_ratio", 10.0)),
                min_board_overlap_ratio=float(component_cfg.get("min_board_overlap_ratio", 0.85)),
            )
        )
        matchers[label] = matcher

    return specs, matchers


def build_detector(config: dict, source: str) -> BoardFirstDetector:
    """Create the full board-first detector from the YAML config."""
    board_cfg = config["board"]
    tracking_cfg = config["tracking"]
    templates_cfg = config["templates"]
    prepared_bank = _load_prepared_template_bank(config)
    template_rotation_turns = int(templates_cfg.get("rotation_turns", 0))

    reference_boards: list[np.ndarray] = []
    board_dirs: list[Path] = []
    if prepared_bank is not None:
        board_dirs.append(prepared_bank.board_dir)
    board_dirs.extend(path for path in _existing_directories(templates_cfg["board_dirs"]) if path not in board_dirs)
    if board_dirs:
        board_paths: list[Path] = []
        for board_dir in board_dirs:
            board_paths.extend(path for path in sorted(board_dir.glob("*")) if path.is_file())
        board_paths = sample_evenly(board_paths, min(len(board_paths), int(board_cfg["max_reference_templates"])))
        reference_boards = _rotate_template_bank([load_bgr(path) for path in board_paths], template_rotation_turns)
        LOGGER.info(
            "Loaded %d board reference images from %s",
            len(reference_boards),
            [str(path) for path in board_dirs],
        )
    else:
        LOGGER.warning("No board reference directory found in %s", templates_cfg["board_dirs"])

    board_locator = _build_board_template_locator(config, board_dirs, template_rotation_turns)
    output_size = (
        tuple(int(value) for value in prepared_bank.canonical_size)
        if prepared_bank is not None
        else (int(board_cfg["output_size"][0]), int(board_cfg["output_size"][1]))
    )
    expected_aspect_ratio = max(output_size[0] / output_size[1], output_size[1] / output_size[0])

    localizer = BoardLocalizer(
        cfg=BoardWarpConfig(
            output_size=output_size,
            blur_ksize=int(board_cfg["blur_ksize"]),
            canny_t1=int(board_cfg["canny_t1"]),
            canny_t2=int(board_cfg["canny_t2"]),
            min_area_ratio=float(board_cfg["min_area_ratio"]),
            max_area_ratio=float(board_cfg["max_area_ratio"]),
            min_rectangularity=float(board_cfg["min_rectangularity"]),
            expected_aspect_ratio=float(expected_aspect_ratio),
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
            verify_resize_width=int(board_cfg.get("verify_resize_width", 300)),
            min_objectness_score=float(board_cfg.get("min_objectness_score", 0.30)),
            geometry_weight=float(board_cfg.get("geometry_weight", 0.45)),
            verify_weight=float(board_cfg.get("verify_weight", 0.25)),
            objectness_weight=float(board_cfg.get("objectness_weight", 0.30)),
        ),
        reference_boards=reference_boards,
    )

    specs, matchers = _component_specs_and_matchers(config, prepared_bank, template_rotation_turns)
    LOGGER.info("Enabled component matchers: %s", sorted(matchers.keys()))

    temporal_min_hits = int(tracking_cfg["temporal_min_hits"]) if source in {"webcam", "video"} else 1
    enable_tracking = source in {"webcam", "video"}
    board_template_cfg = config.get("board_template", {})
    template_refresh_interval = int(board_template_cfg.get("refresh_interval", 5)) if enable_tracking else 1

    return BoardFirstDetector(
        localizer=localizer,
        component_matchers=matchers,
        component_specs=specs,
        board_locator=board_locator,
        cfg=BoardFirstConfig(
            temporal_window=int(tracking_cfg["temporal_window"]),
            temporal_min_hits=temporal_min_hits,
            max_missing_frames=int(tracking_cfg["max_missing_frames"]),
            template_refresh_interval=template_refresh_interval,
            hint_accept_score=float(tracking_cfg.get("hint_accept_score", 0.58)),
            enable_tracking=enable_tracking,
        ),
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.logging)
    config = _apply_source_profile(load_yaml(args.config), args.source)

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
