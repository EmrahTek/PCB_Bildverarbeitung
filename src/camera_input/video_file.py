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
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

from src.camera_input.base import FrameSource
from src.utils.types import FrameMeta

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoFileConfig:
    """Configuration for VideoFileSource."""
    path: Path
    loop: bool = False
    resize_width: int | None = None
    resize_height: int | None = None
    stride: int = 1  # 1=every frame, 2=skip 1 decode 1, ...


class VideoFileSource(FrameSource):
    """
    Video-file frame source for deterministic tests and debugging.

    Features:
    - Optional loop playback
    - Optional stride (frame skipping via grab())
    - Optional resize for faster processing/rendering
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

        # ---------------------------------------------------------
        # 1) Optional frame skipping (stride) to speed up playback.
        #    grab() advances the stream without decoding the frame.
        #    Example: stride=2 -> skip 1 frame, decode 1 frame.
        # ---------------------------------------------------------
        stride = max(1, int(self._cfg.stride))
        for _ in range(stride - 1):
            if not self._cap.grab():
                # End of stream while skipping
                if self._cfg.loop:
                    self._cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                    self._frame_id = 0
                    if not self._cap.grab():
                        return None, None
                else:
                    return None, None

        # Decode one frame
        ok, frame = self._cap.read()
        if not ok or frame is None:
            if self._cfg.loop:
                # Restart video and try again
                self._cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                self._frame_id = 0
                ok2, frame2 = self._cap.read()
                if not ok2 or frame2 is None:
                    return None, None
                frame = frame2
            else:
                return None, None

        # ---------------------------------------------------------
        # 2) Optional resize for faster rendering/processing.
        #    If only width or height is given, keep aspect ratio.
        # ---------------------------------------------------------
        if self._cfg.resize_width is not None or self._cfg.resize_height is not None:
            h, w = frame.shape[:2]

            if self._cfg.resize_width is None:
                # Keep aspect ratio based on height
                scale = float(self._cfg.resize_height) / float(h)
                new_w = int(round(w * scale))
                new_h = int(round(self._cfg.resize_height))
            elif self._cfg.resize_height is None:
                # Keep aspect ratio based on width
                scale = float(self._cfg.resize_width) / float(w)
                new_w = int(round(self._cfg.resize_width))
                new_h = int(round(h * scale))
            else:
                new_w = int(self._cfg.resize_width)
                new_h = int(self._cfg.resize_height)

            frame = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)

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