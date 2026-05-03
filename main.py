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
from src.camera_input.ids import IDSCameraConfig, IDSCameraSource, discover_video_devices, pyueye_available
from src.camera_input.image import ImageFileConfig, ImageFileSource, ImageFolderConfig, ImageFolderSource
from src.camera_input.picamera import PiCameraConfig, PiCameraSource, picamera2_available
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


def _opencv_backend_available(backend_name: str) -> bool:
    backend_id = getattr(cv, backend_name, None)
    if backend_id is None or not hasattr(cv, "videoio_registry"):
        return False
    try:
        return bool(cv.videoio_registry.hasBackend(backend_id))
    except cv.error:
        return False


def _print_video_devices() -> None:
    devices = discover_video_devices()
    if devices:
        print("Video devices:")
        for device in devices:
            print(f"  {device.path}\tindex={device.index}\tname={device.name or 'unknown'}")
    else:
        print("Video devices: none found under /sys/class/video4linux")

    print("Backends:")
    print(f"  OpenCV CAP_V4L2 available: {_opencv_backend_available('CAP_V4L2')}")
    print(f"  OpenCV CAP_UEYE available: {_opencv_backend_available('CAP_UEYE')}")
    print(f"  pyueye installed: {pyueye_available()}")
    print(f"  Picamera2 installed: {picamera2_available()}")


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
    if args.source == "ids":
        return IDSCameraSource(
            IDSCameraConfig(
                index=args.camera_index,
                device=args.camera_device,
                width=args.width,
                height=args.height,
                target_fps=args.camera_fps,
                backend=args.camera_backend,
                use_mjpg=not args.disable_mjpg,
                buffer_size=args.camera_buffer,
                allow_unverified_opencv=args.ids_allow_unverified_opencv,
            )
        )

    if args.source == "webcam":
        return WebcamSource(
            WebcamConfig(
                index=args.camera_index,
                device=args.camera_device,
                width=args.width,
                height=args.height,
                target_fps=args.camera_fps,
                backend=args.camera_backend,
                use_mjpg=not args.disable_mjpg,
                buffer_size=args.camera_buffer,
                source_name="webcam",
            )
        )

    if args.source == "picamera":
        return PiCameraSource(
            PiCameraConfig(
                camera_num=args.camera_index,
                width=args.width,
                height=args.height,
                target_fps=args.camera_fps,
                buffer_count=args.camera_buffer,
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


def _expand_relative_roi_values(
    values: tuple[float, float, float, float],
    expansion: float,
) -> tuple[float, float, float, float]:
    """Expand an ROI around its center while keeping it inside canonical bounds."""
    x1, y1, x2, y2 = (float(value) for value in values)
    amount = max(0.0, float(expansion))
    if amount <= 0.0:
        return x1, y1, x2, y2
    dx = (x2 - x1) * amount
    dy = (y2 - y1) * amount
    return (
        max(0.0, x1 - dx),
        max(0.0, y1 - dy),
        min(1.0, x2 + dx),
        min(1.0, y2 + dy),
    )


def _trim_relative_roi_values(
    values: tuple[float, float, float, float],
    *,
    left: float = 0.0,
    right: float = 0.0,
    top: float = 0.0,
    bottom: float = 0.0,
) -> tuple[float, float, float, float]:
    """Trim ROI edges by fractions of the ROI size while preserving valid bounds."""
    x1, y1, x2, y2 = (float(value) for value in values)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    nx1 = x1 + width * max(0.0, float(left))
    nx2 = x2 - width * max(0.0, float(right))
    ny1 = y1 + height * max(0.0, float(top))
    ny2 = y2 - height * max(0.0, float(bottom))
    if nx2 <= nx1:
        nx1, nx2 = x1, x2
    if ny2 <= ny1:
        ny1, ny2 = y1, y2
    return (
        max(0.0, min(1.0, nx1)),
        max(0.0, min(1.0, ny1)),
        max(0.0, min(1.0, nx2)),
        max(0.0, min(1.0, ny2)),
    )


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
            min_score_margin=float(board_template_cfg.get("min_score_margin", 0.0)),
            second_best_iou_threshold=float(board_template_cfg.get("second_best_iou_threshold", 0.45)),
            preprocess_mode=str(board_template_cfg.get("preprocess_mode", "default")),
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
                min_score_margin=float(component_cfg.get("min_score_margin", 0.0)),
                second_best_iou_threshold=float(component_cfg.get("second_best_iou_threshold", 0.45)),
                preprocess_mode=str(component_cfg.get("preprocess_mode", "default")),
            ),
        )
        roi = (
            prepared_bank.component_rois.get(label, tuple(component_cfg["roi"]))
            if prepared_bank is not None and use_prepared_rois
            else tuple(component_cfg["roi"])
        )
        roi = _expand_relative_roi_values(
            tuple(float(value) for value in roi),
            float(component_cfg.get("search_roi_expansion", 0.0)),
        )
        layout_roi = None
        if prepared_bank is not None and label in prepared_bank.component_rois:
            layout_roi_values = _expand_relative_roi_values(
                prepared_bank.component_rois[label],
                float(component_cfg.get("layout_roi_expansion", 0.0)),
            )
            layout_roi_values = _trim_relative_roi_values(
                layout_roi_values,
                left=float(component_cfg.get("layout_roi_left_trim", 0.0)),
                right=float(component_cfg.get("layout_roi_right_trim", 0.0)),
                top=float(component_cfg.get("layout_roi_top_trim", 0.0)),
                bottom=float(component_cfg.get("layout_roi_bottom_trim", 0.0)),
            )
            layout_roi = RelativeROI(
                float(layout_roi_values[0]),
                float(layout_roi_values[1]),
                float(layout_roi_values[2]),
                float(layout_roi_values[3]),
            )
        elif "layout_roi" in component_cfg:
            layout_roi_values = _expand_relative_roi_values(
                tuple(component_cfg["layout_roi"]),
                float(component_cfg.get("layout_roi_expansion", 0.0)),
            )
            layout_roi_values = _trim_relative_roi_values(
                layout_roi_values,
                left=float(component_cfg.get("layout_roi_left_trim", 0.0)),
                right=float(component_cfg.get("layout_roi_right_trim", 0.0)),
                top=float(component_cfg.get("layout_roi_top_trim", 0.0)),
                bottom=float(component_cfg.get("layout_roi_bottom_trim", 0.0)),
            )
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
                min_warp_quality_score=float(component_cfg.get("min_warp_quality_score", 0.0)),
                layout_fallback_min_warp_quality=float(component_cfg.get("layout_fallback_min_warp_quality", 0.0)),
                layout_fallback_min_match_score=float(component_cfg.get("layout_fallback_min_match_score", 0.0)),
                min_visibility_score=float(component_cfg.get("min_visibility_score", 0.0)),
                layout_fallback_min_visibility_score=float(component_cfg.get("layout_fallback_min_visibility_score", 0.0)),
                visibility_weight=float(component_cfg.get("visibility_weight", 0.0)),
                warp_quality_weight=float(component_cfg.get("warp_quality_weight", 0.0)),
                keep_score_threshold=float(component_cfg.get("keep_score_threshold", 0.0)),
                keep_min_visibility_score=float(component_cfg.get("keep_min_visibility_score", 0.0)),
                local_search_expansion=float(component_cfg.get("local_search_expansion", 0.55)),
                track_max_missing=int(component_cfg.get("track_max_missing", 2)),
                track_smoothing_alpha=float(component_cfg.get("track_smoothing_alpha", 0.55)),
                position_prior_weight=float(component_cfg.get("position_prior_weight", 0.0)),
                min_position_prior_acquire=float(component_cfg.get("min_position_prior_acquire", 0.0)),
                min_position_prior_keep=float(component_cfg.get("min_position_prior_keep", 0.0)),
                persistence_decay=float(component_cfg.get("persistence_decay", 0.88)),
                visibility_upscale=float(component_cfg.get("visibility_upscale", 1.0)),
                layout_anchor=bool(component_cfg.get("layout_anchor", False)),
                output_bbox_pad_left=float(component_cfg.get("output_bbox_pad_left", 0.0)),
                output_bbox_pad_right=float(component_cfg.get("output_bbox_pad_right", 0.0)),
                output_bbox_pad_top=float(component_cfg.get("output_bbox_pad_top", 0.0)),
                output_bbox_pad_bottom=float(component_cfg.get("output_bbox_pad_bottom", 0.0)),
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
            min_warp_quality_score=float(board_cfg.get("min_warp_quality_score", 0.38)),
            min_tracked_warp_quality_score=float(board_cfg.get("min_tracked_warp_quality_score", 0.32)),
            min_pcb_structure_score=float(board_cfg.get("min_pcb_structure_score", 0.12)),
            min_canonical_structure_score=float(board_cfg.get("min_canonical_structure_score", 0.0)),
            min_edge_grid_score=float(board_cfg.get("min_edge_grid_score", 0.0)),
            min_tightness_score=float(board_cfg.get("min_tightness_score", 0.35)),
            refine_pad_x_ratio=float(board_cfg.get("refine_pad_x_ratio", 0.035)),
            refine_pad_y_ratio=float(board_cfg.get("refine_pad_y_ratio", 0.045)),
            refine_pad_right_ratio=float(board_cfg.get("refine_pad_right_ratio", 0.045)),
            refine_pad_right_connector_ratio=float(board_cfg.get("refine_pad_right_connector_ratio", 0.078)),
            refine_connector_score_threshold=float(board_cfg.get("refine_connector_score_threshold", 0.26)),
            max_skin_ratio=float(board_cfg.get("max_skin_ratio", 0.18)),
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

    live_sources = {"webcam", "video", "ids", "picamera"}
    temporal_window = int(tracking_cfg["temporal_window"]) if source in live_sources else 1
    temporal_min_hits = int(tracking_cfg["temporal_min_hits"]) if source in live_sources else 1
    enable_tracking = source in live_sources
    board_template_cfg = config.get("board_template", {})
    template_refresh_interval = int(board_template_cfg.get("refresh_interval", 5)) if enable_tracking else 1

    return BoardFirstDetector(
        localizer=localizer,
        component_matchers=matchers,
        component_specs=specs,
        board_locator=board_locator,
        cfg=BoardFirstConfig(
            temporal_window=temporal_window,
            temporal_min_hits=temporal_min_hits,
            max_missing_frames=int(tracking_cfg["max_missing_frames"]),
            template_refresh_interval=template_refresh_interval,
            hint_accept_score=float(tracking_cfg.get("hint_accept_score", 0.58)),
            enable_tracking=enable_tracking,
            max_pose_reuse_frames=int(tracking_cfg.get("max_pose_reuse_frames", 2)),
            reuse_min_warp_quality=float(tracking_cfg.get("reuse_min_warp_quality", 0.48)),
            reuse_score_decay=float(tracking_cfg.get("reuse_score_decay", 0.92)),
            board_smoothing_alpha=float(tracking_cfg.get("board_smoothing_alpha", 0.55)),
            board_smoothing_min_quality=float(tracking_cfg.get("board_smoothing_min_quality", 0.58)),
            board_smoothing_max_shift=float(tracking_cfg.get("board_smoothing_max_shift", 0.055)),
            board_pose_max_area_growth=float(tracking_cfg.get("board_pose_max_area_growth", 0.22)),
            board_pose_max_quality_drop=float(tracking_cfg.get("board_pose_max_quality_drop", 0.16)),
            board_pose_max_tightness_drop=float(tracking_cfg.get("board_pose_max_tightness_drop", 0.20)),
            board_pose_quality_margin=float(tracking_cfg.get("board_pose_quality_margin", 0.05)),
            board_pose_reuse_decay=float(tracking_cfg.get("board_pose_reuse_decay", 0.98)),
            board_bbox_pad_left=float(tracking_cfg.get("board_bbox_pad_left", 0.0)),
            board_bbox_pad_right=float(tracking_cfg.get("board_bbox_pad_right", 0.0)),
            board_bbox_pad_top=float(tracking_cfg.get("board_bbox_pad_top", 0.0)),
            board_bbox_pad_bottom=float(tracking_cfg.get("board_bbox_pad_bottom", 0.0)),
        ),
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.logging)
    if args.list_video_devices:
        _print_video_devices()
        return

    config = _apply_source_profile(load_yaml(args.config), args.source)
    LOGGER.info("Using source profile: %s", args.source)

    if args.camera_open_check:
        if args.source not in {"webcam", "ids", "picamera"}:
            raise ValueError("--camera-open-check is only valid with --source webcam, --source ids, or --source picamera")
        source = build_source(args)
        try:
            try:
                source.open()
                frame, meta = source.read()
                if frame is None or meta is None:
                    raise RuntimeError("Camera opened, but no frame could be read.")
                LOGGER.info(
                    "Camera open check OK: source=%s frame_shape=%s dtype=%s",
                    meta.source,
                    frame.shape,
                    frame.dtype,
                )
                if args.save_first_frame is not None:
                    args.save_first_frame.parent.mkdir(parents=True, exist_ok=True)
                    if not cv.imwrite(str(args.save_first_frame), frame):
                        raise RuntimeError(f"Failed to save first frame to {args.save_first_frame}")
                    LOGGER.info("Saved first camera frame to %s", args.save_first_frame)
            except RuntimeError as exc:
                LOGGER.error("Camera open check failed: %s", exc)
                raise SystemExit(2) from None
        finally:
            source.release()
        return

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
    try:
        pipeline.run(
            source,
            debug=args.debug,
            headless=args.headless,
            max_frames=args.max_frames,
            wait_ms=args.wait_ms,
        )
    except RuntimeError as exc:
        if args.source in {"webcam", "ids", "picamera"}:
            LOGGER.error("Camera runtime failed: %s", exc)
            raise SystemExit(2) from None
        raise


if __name__ == "__main__":
    main()
