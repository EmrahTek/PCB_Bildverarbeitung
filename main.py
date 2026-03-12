from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2 as cv
import numpy as np

from src.app.cli import parse_args
from src.app.pipeline import Pipeline
from src.camera_input.image import ImageFileSource, ImageFileConfig, ImageFolderSource, ImageFolderConfig
from src.camera_input.video_file import VideoFileSource, VideoFileConfig
from src.camera_input.webcam import WebcamSource, WebcamConfig
from src.detection_logic.template_match import (
    CompositeTemplateMatcher,
    ComponentTemplateMatcher,
    TemplateMatchConfig,
)
from src.logging.setup import setup_logging
from src.preprocessing.geometry import BoardWarpConfig, warp_board
from src.utils.io import load_templates

LOGGER = logging.getLogger(__name__)


class BoardWarpPreprocessor:
    """Optional board warp. Keep disabled for live camera unless explicitly requested."""

    def __init__(self, cfg: BoardWarpConfig) -> None:
        self._cfg = cfg

    def process(self, frame: np.ndarray) -> np.ndarray:
        warped, _H = warp_board(frame, self._cfg)
        return warped if warped is not None else frame


class ResizePreprocessor:
    """Resize while keeping aspect ratio. Prevents very large frames from slowing detection."""

    def __init__(self, width: int) -> None:
        self._width = int(width)

    def process(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._width <= 0 or w <= self._width:
            return frame
        scale = self._width / float(w)
        new_h = max(1, int(round(h * scale)))
        return cv.resize(frame, (self._width, new_h), interpolation=cv.INTER_AREA)


class ComposePreprocessor:
    """Apply preprocessors in sequence."""

    def __init__(self, steps: list[object]) -> None:
        self._steps = steps

    def process(self, frame: np.ndarray) -> np.ndarray:
        out = frame
        for step in self._steps:
            out = step.process(out)
        return out


@dataclass(frozen=True)
class ComponentSpec:
    label: str
    folder_candidates: tuple[str, ...]
    template_limit_live: int
    template_limit_images: int
    enable_rot90_for_images: bool = True


def _source_is_live(source: str) -> bool:
    return source.lower() in ("webcam", "video")


def _get_matcher_profile(args) -> str:
    value = getattr(args, "matcher_profile", "balanced")
    return str(value).lower() if value else "balanced"


def _should_enable_board_warp(args) -> bool:
    """
    Keep the camera image natural by default.
    Board warp is only enabled when the old CLI flag --warp-board is explicitly present.

    This intentionally ignores newer 'disable_board_warp' defaults because some project states
    set that flag to False by default, which would otherwise distort live camera output.
    """
    return bool(getattr(args, "warp_board", False))


def _resolve_template_dir(candidates: Iterable[str]) -> Path | None:
    for candidate in candidates:
        p = Path(candidate)
        if p.exists() and p.is_dir():
            return p
    return None


def _augment_templates(templates: list[np.ndarray], *, rotations: tuple[int, ...]) -> list[np.ndarray]:
    if not templates:
        return []

    out: list[np.ndarray] = []
    for t in templates:
        out.append(t)
        for rot in rotations:
            if rot == 180:
                out.append(cv.rotate(t, cv.ROTATE_180))
            elif rot == 90:
                out.append(cv.rotate(t, cv.ROTATE_90_CLOCKWISE))
            elif rot == 270:
                out.append(cv.rotate(t, cv.ROTATE_90_COUNTERCLOCKWISE))
    return out


def _build_component_specs() -> list[ComponentSpec]:
    return [
        ComponentSpec(
            label="ESP32",
            folder_candidates=(
                "assets/templates/esp32_module",
                "assets/templates/ESP32",
                "assets/templates/esp32",
            ),
            template_limit_live=2,
            template_limit_images=4,
            enable_rot90_for_images=True,
        ),
        ComponentSpec(
            label="JST",
            folder_candidates=(
                "assets/templates/jst_connector",
                "assets/templates/JST",
                "assets/templates/jst",
            ),
            template_limit_live=2,
            template_limit_images=4,
            enable_rot90_for_images=True,
        ),
        ComponentSpec(
            label="USB",
            folder_candidates=(
                "assets/templates/usb_port",
                "assets/templates/USB",
                "assets/templates/usb",
            ),
            template_limit_live=2,
            template_limit_images=4,
            enable_rot90_for_images=True,
        ),
        ComponentSpec(
            label="RESET",
            folder_candidates=(
                "assets/templates/reset_button",
                "assets/templates/RESET",
                "assets/templates/reset",
            ),
            template_limit_live=2,
            template_limit_images=4,
            enable_rot90_for_images=True,
        ),
    ]


def _make_config_for_component(
    label: str,
    *,
    source: str,
    profile: str,
    use_board_warp: bool,
) -> TemplateMatchConfig:
    src = source.lower()
    live = _source_is_live(src)

    # Live mode: fewer scales, ROI tracking on, faster matching.
    if live:
        base_scales = {
            "fast": (0.70, 0.85, 1.00, 1.15),
            "balanced": (0.60, 0.75, 0.90, 1.05, 1.20),
            "accurate": (0.55, 0.70, 0.85, 1.00, 1.15, 1.30),
        }.get(profile, (0.60, 0.75, 0.90, 1.05, 1.20))

        threshold_map = {
            "ESP32": 0.76,
            "JST": 0.70,
            "USB": 0.72,
            "RESET": 0.67,
        }
        use_edges_map = {
            "ESP32": False,
            "JST": True,
            "USB": True,
            "RESET": True,
        }
        return TemplateMatchConfig(
            label=label,
            score_threshold=threshold_map.get(label, 0.72),
            scales=base_scales,
            top_k=1,
            nms_iou_threshold=0.20,
            max_detections=3,
            max_candidates_per_scale=1,
            use_edges=use_edges_map.get(label, False),
            edge_weight=0.30 if use_edges_map.get(label, False) else 0.0,
            use_tracking=True,
            tracking_expansion=1.8,
            force_full_frame_every_n=12,
        )

    # Offline images: slightly wider search, but still no image distortion unless explicitly requested.
    base_scales = {
        "fast": (0.55, 0.75, 0.95, 1.15),
        "balanced": (0.50, 0.65, 0.80, 0.95, 1.10, 1.25),
        "accurate": (0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.35),
    }.get(profile, (0.50, 0.65, 0.80, 0.95, 1.10, 1.25))

    if use_board_warp:
        # If someone explicitly enables warp for offline tuning, scales can be tighter.
        base_scales = {
            "fast": (0.80, 0.95, 1.10),
            "balanced": (0.70, 0.85, 1.00, 1.15),
            "accurate": (0.65, 0.80, 0.95, 1.10, 1.25),
        }.get(profile, (0.70, 0.85, 1.00, 1.15))

    threshold_map = {
        "ESP32": 0.72,
        "JST": 0.67,
        "USB": 0.69,
        "RESET": 0.64,
    }
    use_edges_map = {
        "ESP32": False,
        "JST": True,
        "USB": True,
        "RESET": True,
    }
    return TemplateMatchConfig(
        label=label,
        score_threshold=threshold_map.get(label, 0.68),
        scales=base_scales,
        top_k=1,
        nms_iou_threshold=0.25,
        max_detections=3,
        max_candidates_per_scale=1,
        use_edges=use_edges_map.get(label, False),
        edge_weight=0.30 if use_edges_map.get(label, False) else 0.0,
        use_tracking=False,
        tracking_expansion=1.6,
        force_full_frame_every_n=1,
    )


def _build_detector(args) -> CompositeTemplateMatcher:
    source = args.source.lower()
    profile = _get_matcher_profile(args)
    use_board_warp = _should_enable_board_warp(args)
    live = _source_is_live(source)

    matchers: list[ComponentTemplateMatcher] = []
    for spec in _build_component_specs():
        template_dir = _resolve_template_dir(spec.folder_candidates)
        if template_dir is None:
            LOGGER.info("No template directory found for %s. Skipping.", spec.label)
            continue

        template_limit = spec.template_limit_live if live else spec.template_limit_images
        templates = load_templates(template_dir, limit=template_limit)
        if not templates:
            LOGGER.info("Template directory for %s is empty: %s", spec.label, template_dir)
            continue

        if live:
            # Keep live mode fast. No 90-degree augmentation by default.
            rotations = (180,)
        else:
            rotations = (180, 90, 270) if spec.enable_rot90_for_images else (180,)

        templates = _augment_templates(templates, rotations=rotations)
        cfg = _make_config_for_component(
            spec.label,
            source=source,
            profile=profile,
            use_board_warp=use_board_warp,
        )
        matchers.append(ComponentTemplateMatcher(templates, cfg))
        LOGGER.info(
            "Loaded detector for %s from %s (templates=%d, live=%s, profile=%s)",
            spec.label,
            template_dir,
            len(templates),
            live,
            profile,
        )

    if not matchers:
        raise FileNotFoundError(
            "No component templates were found. Expected folders such as "
            "assets/templates/esp32_module, assets/templates/jst_connector, "
            "assets/templates/usb_port, assets/templates/reset_button."
        )

    return CompositeTemplateMatcher(matchers)


def build_source(args):
    src = args.source.lower()

    if src == "webcam":
        return WebcamSource(
            WebcamConfig(
                index=args.camera_index,
                width=getattr(args, "width", None),
                height=getattr(args, "height", None),
                target_fps=getattr(args, "camera_fps", None),
            )
        )

    if src == "video":
        if args.video_path is None:
            raise ValueError("--video-path is required when --source video")
        return VideoFileSource(
            VideoFileConfig(
                path=args.video_path,
                loop=args.loop,
                resize_width=getattr(args, "video_resize_width", None),
                resize_height=getattr(args, "video_resize_height", None),
                stride=getattr(args, "video_stride", 1),
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


def main() -> None:
    args = parse_args()
    setup_logging(args.logging)
    LOGGER.info("App starting with args=%s", vars(args))

    detector = _build_detector(args)

    preprocessors: list[object] = []
    if getattr(args, "proc_resize_width", None) is not None:
        preprocessors.append(ResizePreprocessor(int(args.proc_resize_width)))

    # Keep images and live camera natural by default. Warp only if explicitly requested.
    if _should_enable_board_warp(args):
        preprocessors.append(
            BoardWarpPreprocessor(
                BoardWarpConfig(
                    output_size=(480, 960),
                    canny_t1=40,
                    canny_t2=140,
                    min_area_ratio=0.08,
                    approx_eps_ratio=0.02,
                )
            )
        )
        LOGGER.info("Board warp explicitly enabled.")
    else:
        LOGGER.info("Board warp disabled. Original frame geometry is preserved.")

    preprocessor = ComposePreprocessor(preprocessors) if preprocessors else None

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
