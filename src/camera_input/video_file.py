# deterministische Tests mit Video-Datei
# Deterministische Frame-Quelle aus Video-Datei (Pflicht für Tests & Reproduzierbarkeit).
"""
video_file.py

This module implements a deterministic FrameSource backed by a video file.
It is critical for reproducible testing and debugging, allowing the pipeline
to run without a live camera.

Inputs:
- Path to a video file
- loop flag to replay the video

Outputs:
- Frames as NumPy arrays (BGR) and FrameMeta objects 

Zu implementierende Funktionen / Klassen

class VideoFileSource(FrameSource):

    __init__(path: Path, loop: bool = False)

    open()

    read() (end-of-file handling)

    close()

    OpenCV VideoCapture also supports video files:
https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np

from src.camera_input.base import FrameSource
from src.utils.types import FrameMeta

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoFileConfig:
    path: Path
    loop: bool = False
    resize_width: int | None = None
    resize_height: int | None = None
    stride: int = 1


class VideoFileSource(FrameSource):
    """Video-file-based frame source for deterministic playback."""

    def __init__(self, cfg: VideoFileConfig) -> None:
        self._cfg = cfg
        self._cap: Optional[cv.VideoCapture] = None
        self._frame_id = 0

    def open(self) -> None:
        if not self._cfg.path.exists():
            raise FileNotFoundError(f"Video file not found: {self._cfg.path}")
        self._cap = cv.VideoCapture(str(self._cfg.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self._cfg.path}")
        LOGGER.info("Video opened: %s", self._cfg.path)

    def read(self) -> tuple[Optional[np.ndarray], Optional[FrameMeta]]:
        if self._cap is None:
            raise RuntimeError("VideoFileSource.read() called before open().")

        stride = max(1, int(self._cfg.stride))
        for _ in range(stride - 1):
            if not self._cap.grab():
                if not self._cfg.loop:
                    return None, None
                self._cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                self._frame_id = 0
                if not self._cap.grab():
                    return None, None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            if not self._cfg.loop:
                return None, None
            self._cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            self._frame_id = 0
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return None, None

        if self._cfg.resize_width is not None or self._cfg.resize_height is not None:
            frame = self._resize(frame)

        meta = FrameMeta(frame_id=self._frame_id, timestamp_s=time.perf_counter(), source=f"video:{self._cfg.path.name}")
        self._frame_id += 1
        return frame, meta

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._cfg.resize_width is None and self._cfg.resize_height is None:
            return frame
        if self._cfg.resize_width is None:
            scale = float(self._cfg.resize_height) / float(h)
            new_h = int(self._cfg.resize_height)
            new_w = int(round(w * scale))
        elif self._cfg.resize_height is None:
            scale = float(self._cfg.resize_width) / float(w)
            new_w = int(self._cfg.resize_width)
            new_h = int(round(h * scale))
        else:
            new_w = int(self._cfg.resize_width)
            new_h = int(self._cfg.resize_height)
        return cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
