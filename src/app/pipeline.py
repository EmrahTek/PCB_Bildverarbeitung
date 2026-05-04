"""Runtime capture, detection, and rendering loop.

The Pipeline class reads frames from a FrameSource, optionally preprocesses
them, runs the detector, and either renders OpenCV output or runs headlessly.

Python docs:
- dataclasses: https://docs.python.org/3/library/dataclasses.html
- logging: https://docs.python.org/3/library/logging.html
- typing.Protocol: https://docs.python.org/3/library/typing.html#typing.Protocol
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import cv2 as cv
import numpy as np

from src.camera_input.base import FrameSource
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
    """Default preprocessor used when no resize step is requested."""

    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame


@dataclass(frozen=True)
class PipelineConfig:
    window_name: str = "PCB Component Detection"
    exit_key: str = "q"


class Pipeline:
    """Run the full capture -> preprocess -> detect -> render loop."""

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

    def run(
        self,
        source: FrameSource,
        *,
        debug: bool = False,
        headless: bool = False,
        max_frames: int | None = None,
        wait_ms: int = 1,
    ) -> None:
        source.open()
        if not headless:
            cv.namedWindow(self._cfg.window_name, cv.WINDOW_NORMAL)
            cv.resizeWindow(self._cfg.window_name, 960, 540)

        LOGGER.info("Pipeline started: headless=%s debug=%s max_frames=%s", headless, debug, max_frames)

        frame_count = 0
        try:
            while True:
                frame, meta = source.read()
                if frame is None or meta is None:
                    LOGGER.info("End of stream reached.")
                    break

                processed = self._preprocessor.process(frame)
                detections = self._detector.detect(processed)

                if debug and (meta.frame_id % 15 == 0 or meta.source.startswith("image:") or meta.source.startswith("images:")):
                    labels = ", ".join(det.label for det in detections) if detections else "none"
                    best = max((det.score for det in detections), default=0.0)
                    LOGGER.info("frame=%d source=%s labels=%s best=%.3f", meta.frame_id, meta.source, labels, best)

                visualized = draw_detections(processed, detections, debug=debug)
                if not headless:
                    cv.imshow(self._cfg.window_name, visualized)
                    key = cv.waitKey(wait_ms) & 0xFF
                    if key == ord(self._cfg.exit_key):
                        LOGGER.info("Exit key pressed.")
                        break

                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    LOGGER.info("Reached max_frames=%d.", max_frames)
                    break
        finally:
            source.release()
            if not headless:
                cv.destroyWindow(self._cfg.window_name)
            LOGGER.info("Pipeline stopped cleanly.")
