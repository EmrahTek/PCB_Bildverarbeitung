from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
from src.detection_logic.board_detector import BoardFirstHybridDetector

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------
def _filtered_kwargs(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


def _try_make_config(cls: type, candidates: list[dict[str, Any]]) -> Any:
    """
    Try several constructor keyword variants.
    This makes main.py more robust against small config class differences.
    """
    last_error: Exception | None = None

    for kwargs in candidates:
        try:
            return cls(**_filtered_kwargs(kwargs))
        except TypeError as exc:
            last_error = exc

    # last fallback: try default constructor + setattr
    try:
        obj = cls()
        for kwargs in candidates:
            for key, value in _filtered_kwargs(kwargs).items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
        return obj
    except Exception as exc:
        last_error = exc

    raise RuntimeError(f"Could not construct config for {cls.__name__}: {last_error}")


# ---------------------------------------------------------------------
# source builders
# ---------------------------------------------------------------------
def _build_webcam_source(args) -> WebcamSource:
    config = _try_make_config(
        WebcamConfig,
        candidates=[
            {
                "index": args.camera_index,
                "width": args.width,
                "height": args.height,
                "fps": args.camera_fps,
            },
            {
                "camera_index": args.camera_index,
                "width": args.width,
                "height": args.height,
                "fps": args.camera_fps,
            },
        ],
    )
    return WebcamSource(config)


def _build_video_source(args) -> VideoFileSource:
    video_path = Path(args.video_path)

    config = _try_make_config(
        VideoFileConfig,
        candidates=[
            {
                "path": video_path,
                "resize_width": args.video_resize_width,
                "resize_height": args.video_resize_height,
                "stride": args.video_stride,
                "loop": args.loop,
            },
            {
                "video_path": video_path,
                "resize_width": args.video_resize_width,
                "resize_height": args.video_resize_height,
                "stride": args.video_stride,
                "loop": args.loop,
            },
        ],
    )
    return VideoFileSource(config)


def _build_image_file_source(args) -> ImageFileSource:
    image_path = Path(args.image_path)

    config = _try_make_config(
        ImageFileConfig,
        candidates=[
            {"path": image_path},
            {"image_path": image_path},
            {"file_path": image_path},
        ],
    )
    return ImageFileSource(config)


def _build_image_folder_source(args) -> ImageFolderSource:
    images_dir = Path(args.images_dir)

    config = _try_make_config(
        ImageFolderConfig,
        candidates=[
            {
                "directory": images_dir,
                "recursive": args.recursive,
                "loop": args.loop,
            },
            {
                "path": images_dir,
                "recursive": args.recursive,
                "loop": args.loop,
            },
            {
                "folder_path": images_dir,
                "recursive": args.recursive,
                "loop": args.loop,
            },
            {
                "images_dir": images_dir,
                "recursive": args.recursive,
                "loop": args.loop,
            },
        ],
    )
    return ImageFolderSource(config)


def build_source(args):
    if args.source == "webcam":
        return _build_webcam_source(args)

    if args.source == "video":
        if not args.video_path:
            raise ValueError("--video-path must be provided when --source video")
        return _build_video_source(args)

    if args.source == "image":
        if not args.image_path:
            raise ValueError("--image-path must be provided when --source image")
        return _build_image_file_source(args)

    if args.source == "images":
        if not args.images_dir:
            raise ValueError("--images-dir must be provided when --source images")
        return _build_image_folder_source(args)

    raise ValueError(f"Unsupported source: {args.source}")


# ---------------------------------------------------------------------
# matcher / detector builders
# ---------------------------------------------------------------------
def _make_template_config(args) -> TemplateMatchConfig:
    """
    Build TemplateMatchConfig in a defensive way.
    We only set attributes if they exist in your local class.
    """
    cfg = TemplateMatchConfig()

    is_live = args.source in {"webcam", "video"}

    common_updates = {
        "profile": getattr(args, "matcher_profile", "balanced"),
        "use_clahe": True,
        "use_edges": True,
        "score_threshold": 0.66 if not is_live else 0.68,
        "nms_iou_threshold": 0.25,
        "max_detections": 1,
    }

    image_updates = {
        "scale_factors": [0.85, 0.92, 1.00, 1.08, 1.16],
        "search_step": 1,
    }

    live_updates = {
        "scale_factors": [0.92, 1.00, 1.08],
        "search_step": 2,
    }

    updates = common_updates | (live_updates if is_live else image_updates)

    for key, value in updates.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    return cfg


def _load_esp32_templates(args):
    template_dir = Path("assets/templates/esp32_module")
    templates = load_templates(template_dir)

    is_live = args.source in {"webcam", "video"}

    # Live mode: fewer templates -> faster and usually more stable
    if is_live and len(templates) > 4:
        templates = templates[:4]

    LOGGER.info(
        "Loaded ESP32 templates from %s (raw=%d, used=%d, source=%s)",
        template_dir,
        len(load_templates(template_dir)),
        len(templates),
        args.source,
    )
    return templates


def build_detector(args):
    source_mode = "live" if args.source in {"webcam", "video"} else "images"

    esp32_templates = _load_esp32_templates(args)
    esp32_config = _make_template_config(args)
    esp32_matcher = TemplateMatcher(templates=esp32_templates, config=esp32_config)

    detector = BoardFirstHybridDetector(
        board_template_dir=Path("assets/templates/BoardFireBeetle"),
        esp32_matcher=esp32_matcher,
        source_mode=source_mode,
    )

    LOGGER.info(
        "Board-first detector active: geometry/homography for BOARD, "
        "template matching for ESP32 inside warped board."
    )

    return detector


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    setup_logging(args.logging)

    LOGGER.info("App starting with args=%s", vars(args))

    source = build_source(args)
    detector = build_detector(args)

    # IMPORTANT:
    # External board warp preprocessor is intentionally disabled.
    # BoardFirstHybridDetector handles board geometry + warp internally.
    preprocessor = None

    pipeline = Pipeline(
        source=source,
        detector=detector,
        preprocessor=preprocessor,
    )

    try:
        pipeline.run(
            headless=args.headless,
            debug=args.debug,
            max_frames=args.max_frames,
            wait_ms=args.wait_ms,
            log_every_n=args.log_every_n,
        )
    finally:
        LOGGER.info("App finished.")


if __name__ == "__main__":
    main()