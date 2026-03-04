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

class Detector(Protocol):
    """
    Minimal detector protocol.
    Any detector with .detect(frame)->list[Detection] can be plugged in.
    """
    #def detect(self, frame: np.ndarray) -> list[Detection]:

@dataclass(frozen=True)
class PipelineConfig: 
    window_name: str = "PCB Component Detection"
    exit_key: str = "q"

class Pipeline:
    """
    Orchestrates: capture -> detect -> render -> display.
    """
    def __init__(self, detector: Detector, *, cfg: PipelineConfig = PipelineConfig()) -> None:
        self._detector = detector
        self._cfg = cfg
        self._fps = FPSCounter(window_size=30)

    def run(self, source: FrameSource, *, debug: bool = False, headless: bool = False) -> None:
        """
        Run the main loop.

        Args:
            source: FrameSource (webcam/video)
            debug: If True, show detection scores.
            headless: If True, do not open GUI (not used heavily today).
        """
        source.open()
        LOGGER.info("Pipeline started. headless=%s debug=%s", headless, debug)

        try:
            while True:
                frame, meta = source.read()
                if frame is None or meta is None:
                    LOGGER.info("End of stream or read failure. Exiting loop.")

                fps = self._fps.tick()

                # DEtect components
                #detections = self._detector._detector.detect(frame)
                detections = self._detector.detect(frame)
                # Render overlay (copy of the frame)
                vis = draw_detections(frame,detections,fps=fps,debug=debug)

                if not headless:
                    cv.imshow(self._cfg.window_name,vis)

                    # Use waitKey(1) for real-time; read key for exit
                    key = cv.waitKey(1) & 0xFF
                    if key == ord(self._cfg.exit_key):
                        LOGGER.info("Exit key pressed (%s).", self._cfg.exit_key)
                        break

        finally:
            source.release()
            if not headless:
                cv.destroyAllWindows()
            LOGGER.info("Pipeline stopped cleanly.")


