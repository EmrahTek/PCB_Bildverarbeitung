# main.py
from __future__ import annotations

import logging
from pathlib import Path

from src.app.cli import parse_args
from src.app.pipeline import Pipeline
from src.logging.setup import setup_logging

from src.camera_input.webcam import WebcamSource, WebcamConfig
from src.camera_input.video_file import VideoFileSource, VideoFileConfig

from src.utils.io import load_templates
#from src.detection_logic.template_match import TemplateMatcher, TemplateMatchConfig


LOGGER = logging.getLogger(__name__)


def build_source(args):
    """Factory for FrameSource based on CLI args."""
    if args.source == "webcam":
        cfg = WebcamConfig(index=args.camera_index, width=args.width, height=args.height)
        return WebcamSource(cfg)

    if args.source == "video":
        if args.video_path is None:
            raise ValueError("--video-path is required when --source video")
        return VideoFileSource(VideoFileConfig(path=args.video_path, loop=args.loop))

    raise ValueError(f"Unsupported source: {args.source}")


def main() -> None:
    args = parse_args()
    setup_logging(args.logging)

    LOGGER.info("App starting with args=%s", vars(args))

    # Load templates (grayscale)
    template_dir = Path("/home/emrahtek/Schreibtisch/CodeLab/PCB_Bauteilerkennung/assets/templates/esp32_module")
    templates = load_templates(template_dir)

    # Create detector
    detector_cfg = TemplateMatchConfig(
        label="ESP32",
        score_threshold=0.78,  # adjust after first run
        scales=(0.8, 0.9, 1.0, 1.1, 1.2),
        nms_iou_threshold=0.3,
    )
    detector = TemplateMatcher(templates, detector_cfg)

    # Build source and run pipeline
    source = build_source(args)
    pipeline = Pipeline(detector)
    pipeline.run(source, debug=args.debug, headless=args.headless)

    LOGGER.info("App finished.")


if __name__ == "__main__":
    main()