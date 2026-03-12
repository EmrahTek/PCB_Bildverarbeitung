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
import time
from dataclasses import dataclass
from typing import Optional, Protocol

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
    """Return the input frame unchanged."""

    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame


@dataclass(frozen=True)
class PipelineConfig:
    window_name: str = "PCB Component Detection"
    exit_key: str = "q"


class Pipeline:
    """Main orchestration loop: source -> preprocess -> detect -> render."""

    def __init__(
        self,
        detector: DetectorLike,
        *,
        preprocessor: PreprocessorLike | None = None,
        cfg: PipelineConfig = PipelineConfig(),
    ) -> None:
        self._detector = detector
        self._preprocessor = preprocessor if preprocessor is not None else IdentityPreprocessor()
        self._cfg = cfg
        self._fps = FPSCounter(window_size=30)

    def run(
        self,
        source: FrameSource,
        *,
        debug: bool = False,
        headless: bool = False,
        max_frames: int | None = None,
        wait_ms: int = 1,
        log_every_n: int = 30,
    ) -> None:
        source.open()
        if not headless:
            cv.namedWindow(self._cfg.window_name, cv.WINDOW_NORMAL)
            cv.resizeWindow(self._cfg.window_name, 1280, 720)

        frame_count = 0
        LOGGER.info("Pipeline started. headless=%s debug=%s max_frames=%s", headless, debug, max_frames)

        try:
            while True:
                frame, meta = source.read()
                if frame is None or meta is None:
                    LOGGER.info("End of stream reached.")
                    break

                proc = self._preprocessor.process(frame)
                start = time.perf_counter()
                detections = self._detector.detect(proc)
                proc_ms = (time.perf_counter() - start) * 1000.0
                fps = self._fps.tick()

                should_log = debug or (meta.frame_id % max(1, log_every_n) == 0)
                if should_log:
                    best = max((d.score for d in detections), default=0.0)
                    LOGGER.info(
                        "frame=%d source=%s count=%d best=%.3f proc_ms=%.1f fps=%.1f",
                        meta.frame_id,
                        meta.source,
                        len(detections),
                        best,
                        proc_ms,
                        fps,
                    )

                board_quad = getattr(self._detector, "last_board_quad", None)
                vis = draw_detections(proc, detections, fps=fps, debug=debug, board_quad=board_quad)

                if not headless:
                    cv.imshow(self._cfg.window_name, vis)
                    key = cv.waitKey(max(1, wait_ms)) & 0xFF
                    if key == ord(self._cfg.exit_key):
                        LOGGER.info("Exit key pressed: %s", self._cfg.exit_key)
                        break
                else:
                    if wait_ms > 0:
                        time.sleep(wait_ms / 1000.0)

                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    LOGGER.info("Reached max_frames=%d", max_frames)
                    break
        finally:
            source.release()
            if not headless:
                cv.destroyWindow(self._cfg.window_name)
            LOGGER.info("Pipeline stopped cleanly.")
