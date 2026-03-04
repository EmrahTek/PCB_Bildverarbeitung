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
"""

"""
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
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

from src.camera_input.base import FrameSource
from src.utils.types import FrameMeta


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoFileConfig:
    path: Path
    loop: bool = False

class VideoFileSource(FrameSource):
    """
    Video-file frame source for deterministic tests and debugging.
    """

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

        fps = float(self._cap.get(cv.CAP_PROP_FPS))
        LOGGER.info("Video opened: %s (fps=%.2f, loop=%s)", self._cfg.path, fps, self._cfg.loop)

    def read(self) -> Tuple[Optional[np.ndarray], Optional[FrameMeta]]:
        if self._cap is None:
            raise RuntimeError("VideoFileSource.read() called before open().")

        ok, frame = self._cap.read()
        if not ok or frame is None:
            if self._cfg.loop:
                # Restart video
                self._cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                self._frame_id = 0
                ok2, frame2 = self._cap.read()
                if not ok2 or frame2 is None:
                    return None, None
                frame = frame2
            else:
                return None, None

        meta = FrameMeta(
            frame_id=self._frame_id,
            timestamp_s=time.perf_counter(),
            source=f"video:{self._cfg.path.name}",
        )
        self._frame_id += 1
        return frame, meta

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            LOGGER.info("Video released.")

