from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from src.app.cli import parse_args
from src.app.pipeline import Pipeline
from src.camera_input.image import (
    ImageFileConfig,
    ImageFileSource,
    ImageFolderConfig,
    ImageFolderSource,
)
from src.camera_input.video_file import VideoFileConfig, VideoFileSource
from src.camera_input.webcam import WebcamConfig, WebcamSource
from src.detection_logic.board_aware import BoardAwareHybridDetector
from src.detection_logic.feature_match import ORBFeatureMatcher, ORBMatchConfig
from src.detection_logic.template_match import TemplateMatchConfig, TemplateMatcher
from src.logging.setup import setup_logging
from src.preprocessing.geometry import BoardLocalizer, BoardWarpConfig
from src.utils.io import load_templates, load_yaml

LOGGER = logging.getLogger(__name__)


class ResizePreprocessor:
    """Resize frames before detection to reduce load and increase FPS."""

    def __init__(self, width: int) -> None:
        self._width = int(width)

    def process(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self._width:
            return frame
        scale = self._width / float(w)
        new_w = self._width
        new_h = int(round(h * scale))
        return cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)


class ComposePreprocessor:
    """Apply multiple preprocessors in sequence."""

    def __init__(self, steps: list[object]) -> None:
        self._steps = steps

    def process(self, frame: np.ndarray) -> np.ndarray:
        out = frame
        for step in self._steps:
            out = step.process(out)
        return out


def build_source(args):
    src = args.source.lower()

    if src == "webcam":
        return WebcamSource(
            WebcamConfig(
                index=args.camera_index,
                width=args.width,
                height=args.height,
                target_fps=args.camera_fps,
                use_mjpg=True,
                buffer_size=1,
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
        return ImageFileSource(ImageFileConfig(path=args.image_path, loop=args.loop))

    if src == "images":
        if args.images_dir is None:
            raise ValueError("--images-dir is required when --source images")
        return ImageFolderSource(
            ImageFolderConfig(directory=args.images_dir, loop=args.loop, recursive=args.recursive)
        )

    raise ValueError(f"Unsupported source: {args.source}")


def _dict_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_roi(value: Any) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"ROI must have 4 integers, got: {value!r}")
    x, y, w, h = value
    return int(x), int(y), int(w), int(h)


def _find_first_existing_dir(base_dir: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = base_dir / name
        if p.exists() and p.is_dir():
            return p
    return None


def _build_template_cfg(label: str, component_cfg: dict[str, Any], profile_cfg: dict[str, Any]) -> TemplateMatchConfig:
    merged = {**component_cfg, **profile_cfg}
    return TemplateMatchConfig(
        label=label,
        score_threshold=float(merged.get("score_threshold", 0.82)),
        scales=tuple(float(x) for x in merged.get("scales", [0.18, 0.20, 0.22, 0.25, 0.28, 0.30])),
        nms_iou_threshold=float(merged.get("nms_iou_threshold", 0.20)),
        max_candidates_per_scale=int(merged.get("max_candidates_per_scale", 40)),
        max_detections=int(merged.get("max_detections", 5)),
        min_template_size=int(merged.get("min_template_size", 12)),
        top_k=int(merged.get("top_k", 1)) if merged.get("top_k", 1) is not None else None,
        use_clahe=bool(merged.get("use_clahe", True)),
        edge_mode=bool(merged.get("edge_mode", True)),
        blur_ksize=int(merged.get("blur_ksize", 3)),
        search_roi=_as_roi(merged.get("roi")),
    )


def _build_orb_cfg(label: str, component_cfg: dict[str, Any]) -> ORBMatchConfig:
    return ORBMatchConfig(
        label=label,
        nfeatures=int(component_cfg.get("orb_nfeatures", 1000)),
        good_match_ratio=float(component_cfg.get("good_match_ratio", 0.75)),
        min_inliers=int(component_cfg.get("min_inliers", 12)),
        search_roi=_as_roi(component_cfg.get("roi")),
    )


def build_detector(app_cfg: dict[str, Any], args) -> BoardAwareHybridDetector:
    assets_dir = Path(_dict_get(app_cfg, "paths", "templates_dir", default="assets/templates"))
    components = _dict_get(app_cfg, "components", default=[])
    if not components:
        components = [
            {
                "label": "ESP32",
                "enabled": True,
                "template_subdir_candidates": ["esp32_module", "ESP32"],
                "fallback_orb": True,
                "roi": None,
            }
        ]

    profile_name = args.matcher_profile
    profile_cfg_all = _dict_get(app_cfg, "matcher_profiles", default={})

    primary_detectors: dict[str, TemplateMatcher] = {}
    fallback_detectors: dict[str, ORBFeatureMatcher] = {}

    for component in components:
        if not bool(component.get("enabled", True)):
            continue

        label = str(component["label"])
        profile_cfg = profile_cfg_all.get(profile_name, {}).get(label, {})
        candidates = list(component.get("template_subdir_candidates", [label]))
        template_dir = _find_first_existing_dir(assets_dir, candidates)
        if template_dir is None:
            LOGGER.warning("Skipping %s because no template folder exists in %s (tried: %s)", label, assets_dir, candidates)
            continue

        templates = load_templates(template_dir)
        if not templates:
            LOGGER.warning("Skipping %s because template folder is empty: %s", label, template_dir)
            continue

        tm_cfg = _build_template_cfg(label, component, profile_cfg)
        primary_detectors[label] = TemplateMatcher(templates, tm_cfg)

        if bool(component.get("fallback_orb", False)):
            fallback_detectors[label] = ORBFeatureMatcher(templates[0], _build_orb_cfg(label, component))

        LOGGER.info("Loaded detector for %s from %s (templates=%d)", label, template_dir, len(templates))

    if not primary_detectors:
        raise RuntimeError("No component detector could be built. Check assets/templates and config/default.yaml.")

    board_cfg = BoardWarpConfig(
        output_size=tuple(_dict_get(app_cfg, "board", "output_size", default=[800, 600])),
        canny_t1=int(_dict_get(app_cfg, "board", "canny_t1", default=50)),
        canny_t2=int(_dict_get(app_cfg, "board", "canny_t2", default=150)),
        min_area_ratio=float(_dict_get(app_cfg, "board", "min_area_ratio", default=0.10)),
        approx_eps_ratio=float(_dict_get(app_cfg, "board", "approx_eps_ratio", default=0.02)),
        blur_ksize=int(_dict_get(app_cfg, "board", "blur_ksize", default=5)),
        morph_ksize=int(_dict_get(app_cfg, "board", "morph_ksize", default=5)),
    )

    localizer = BoardLocalizer(
        board_cfg,
        enabled=not args.disable_board_warp,
        redetect_interval=int(_dict_get(app_cfg, "board", "redetect_interval", default=5)),
    )

    return BoardAwareHybridDetector(
        localizer=localizer,
        primary_detectors=primary_detectors,
        fallback_detectors=fallback_detectors,
        allow_full_frame_fallback=bool(_dict_get(app_cfg, "board", "allow_full_frame_fallback", default=True)),
        temporal_window=int(_dict_get(app_cfg, "temporal", "window", default=5)),
        temporal_min_hits=int(_dict_get(app_cfg, "temporal", "min_hits", default=3)),
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.logging)

    app_cfg = load_yaml(args.config) if args.config.exists() else {}
    LOGGER.info("Application starting with args=%s", vars(args))

    detector = build_detector(app_cfg, args)

    steps: list[object] = []
    if args.proc_resize_width is not None:
        steps.append(ResizePreprocessor(args.proc_resize_width))

    preprocessor = ComposePreprocessor(steps) if steps else None
    source = build_source(args)
    pipeline = Pipeline(detector=detector, preprocessor=preprocessor)

    pipeline.run(
        source,
        debug=args.debug,
        headless=args.headless,
        max_frames=args.max_frames,
        wait_ms=args.wait_ms,
        log_every_n=max(1, args.log_every_n),
    )

    LOGGER.info("Application finished cleanly.")


if __name__ == "__main__":
    main()
