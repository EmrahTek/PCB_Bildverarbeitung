# cv2.VideoCapture
# Webcam-Quelle via cv2.VideoCapture, mit stabiler Fehlerbehandlung und einstellbarer Auflösung.

"""
webcam.py

This module implements a FrameSource for a live webcam using cv2.VideoCapture.
It handles:
- opening the device
- configuring capture properties (resolution, fps)
- reading frames with robust error handling
- closing the device cleanly

Inputs:
- camera_index (int)
- desired width/height (int)
- optional fps (int)

Outputs:
- Frames as NumPy arrays (BGR) and FrameMeta objects


Zu implementierende Funktionen / Klassen

class WebcamSource(FrameSource):

    __init__(camera_index: int, width: int, height: int, fps: int | None)

    open()

    read()

    close()

    set_capture_properties(cap, width, height, fps) -> None

OpenCV VideoCapture:
https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

VideoCapture properties (CAP_PROP_*):
https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html

"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import cv2 as cv
import numpy as np

from src.camera_input.base import FrameSource
from src.utils.types import FrameMeta

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebcamConfig:
    index: int = 0
    width: int | None = None
    height: int | None = None
    target_fps: int | None = None
    use_mjpg: bool = True
    buffer_size: int = 1


class WebcamSource(FrameSource):
    """OpenCV webcam source with best-effort low-latency settings."""

    def __init__(self, cfg: WebcamConfig) -> None:
        self._cfg = cfg
        self._cap: Optional[cv.VideoCapture] = None
        self._frame_id = 0

    def open(self) -> None:
        self._cap = cv.VideoCapture(self._cfg.index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open webcam index {self._cfg.index}")

        if self._cfg.use_mjpg:
            self._cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
        if self._cfg.width is not None:
            self._cap.set(cv.CAP_PROP_FRAME_WIDTH, float(self._cfg.width))
        if self._cfg.height is not None:
            self._cap.set(cv.CAP_PROP_FRAME_HEIGHT, float(self._cfg.height))
        if self._cfg.target_fps is not None:
            self._cap.set(cv.CAP_PROP_FPS, float(self._cfg.target_fps))
        if hasattr(cv, "CAP_PROP_BUFFERSIZE"):
            self._cap.set(cv.CAP_PROP_BUFFERSIZE, float(self._cfg.buffer_size))

        actual_w = int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self._cap.get(cv.CAP_PROP_FPS))
        LOGGER.info(
            "Webcam opened: index=%d size=%dx%d fps=%.1f mjpg=%s buffer=%d",
            self._cfg.index,
            actual_w,
            actual_h,
            actual_fps,
            self._cfg.use_mjpg,
            self._cfg.buffer_size,
        )

    def read(self) -> tuple[Optional[np.ndarray], Optional[FrameMeta]]:
        if self._cap is None:
            raise RuntimeError("WebcamSource.read() called before open().")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None, None
        meta = FrameMeta(frame_id=self._frame_id, timestamp_s=time.perf_counter(), source=f"webcam:{self._cfg.index}")
        self._frame_id += 1
        return frame, meta

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
