# Orchestrierung: capture -> prepocess -> detect -> render
# Orchestrierung: FrameSource → Preprocessing → Detection → Postprocess → Render → Output/Exit.
"""
pipeline.py

This module orchestrates the end-to-end processing pipeline:
capture -> (optional) board normalization -> preprocessing -> detection -> postprocess -> render.

The pipeline is designed for testability by using dependency injection:
- FrameSource provides frames (webcam or video file)
- preprocessors transform frames
- Detector returns structured detections
- Renderer draws overlays

Inputs:
- FrameSource instance
- Configuration dictionary
- Processing components (preprocessors, detector, renderer)

Outputs:
- Rendered frames (for display or debug saving)
- Logs describing runtime status and detection results

Zu implementierende Funktionen / Klassen

class Pipeline:

    __init__(frame_source, preprocessors, detector, renderer, logger, config)

    process_frame(frame, meta) -> np.ndarray | None

    run() -> None

build_pipeline_from_config(config: dict, args) -> Pipeline

safe_imshow_or_headless(...) (je nach --headless)

OpenCV imshow/waitKey:
https://docs.opencv.org/4.x/dc/d2e/tutorial_py_image_display.html

Designing testable pipelines (search terms):
"dependency injection python pipeline"
"clean architecture python small projects"


"""

# src/app/pipeline.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Optional

import cv2 as cv
import numpy as np

from src.camera_input.base import FrameSource
from src.render.fps import FPSCounter
from src.render.overlay import draw_detections
from src.utils.types import Detection

LOGGER = logging.getLogger(__name__)


class DetectorLike(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        ...


class PreprocessorLike(Protocol):
    def process(self, frame: np.ndarray) -> np.ndarray:
        ...


class IdentityPreprocessor:
    """Default preprocessor: returns frame unchanged."""
    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame


@dataclass(frozen=True)
class PipelineConfig:
    window_name: str = "PCB Component Detection"
    exit_key: str = "q"


class Pipeline:
    """
    capture -> preprocess -> detect -> overlay -> display
    """

    def __init__(
        self,
        detector: DetectorLike,
        *,
        preprocessor: PreprocessorLike | None = None,
        cfg: PipelineConfig = PipelineConfig(),
    ) -> None:
        self._detector = detector
        self._pre = preprocessor if preprocessor is not None else IdentityPreprocessor()
        self._cfg = cfg
        self._fps = FPSCounter(window_size=30)

    def run(
        self,
        source: FrameSource,
        *,
        debug: bool = False,
        headless: bool = False,
        max_frames: int | None = None,
    ) -> None:
        """
        Args:
            headless: If True, no GUI window is opened (useful for tests/CI).
            max_frames: Stop after N frames (useful for tests and quick experiments).
        """
        source.open()
        LOGGER.info("Pipeline started. headless=%s debug=%s max_frames=%s", headless, debug, max_frames)

        frame_count = 0
        try:
            while True:
                frame, meta = source.read()
                if frame is None or meta is None:
                    LOGGER.info("End of stream or read failure. Exiting loop.")
                    break

                fps = self._fps.tick()

                # --- Preprocess ---
                proc = self._pre.process(frame)

                # --- Detect ---
                detections = self._detector.detect(proc)

                # --- Render ---
                vis = draw_detections(proc, detections, fps=fps, debug=debug)

                if not headless:
                    cv.imshow(self._cfg.window_name, vis)
                    key = cv.waitKey(1) & 0xFF
                    if key == ord(self._cfg.exit_key):
                        LOGGER.info("Exit key pressed (%s).", self._cfg.exit_key)
                        break

                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    LOGGER.info("Reached max_frames=%d. Stopping.", max_frames)
                    break

        finally:
            source.release()
            # Create a resizable window and force a reasonable size/position
            if not headless:
                cv.namedWindow(self._cfg.window_name, cv.WINDOW_NORMAL)
                cv.resizeWindow(self._cfg.window_name, 960, 540)  # adjust if you want
                cv.moveWindow(self._cfg.window_name, 50, 50)      # keep on primary screen
            LOGGER.info("Pipeline stopped cleanly.")