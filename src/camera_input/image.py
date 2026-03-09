# src/camera_input/image.py
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
from src.utils.io import list_image_files

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageFileConfig:
    path: Path
    loop: bool = False


class ImageFileSource(FrameSource):
    """FrameSource returning a single image as a frame (once or loop)."""

    def __init__(self, cfg: ImageFileConfig) -> None:
        self._cfg = cfg
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._done = False

    def open(self) -> None:
        if not self._cfg.path.exists():
            raise FileNotFoundError(f"Image not found: {self._cfg.path}")

        img = cv.imread(str(self._cfg.path), cv.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image: {self._cfg.path}")

        self._frame = img
        self._frame_id = 0
        self._done = False
        LOGGER.info("ImageFile opened: %s (loop=%s)", self._cfg.path, self._cfg.loop)

    def read(self) -> Tuple[Optional[np.ndarray], Optional[FrameMeta]]:
        if self._frame is None:
            raise RuntimeError("ImageFileSource.read() called before open().")

        if self._done and not self._cfg.loop:
            return None, None

        meta = FrameMeta(
            frame_id=self._frame_id,
            timestamp_s=time.perf_counter(),
            source=f"image:{self._cfg.path.name}",
        )
        self._frame_id += 1
        self._done = True
        return self._frame.copy(), meta

    def release(self) -> None:
        self._frame = None
        LOGGER.info("ImageFile released.")


@dataclass(frozen=True)
class ImageFolderConfig:
    directory: Path
    loop: bool = False
    recursive: bool = False


class ImageFolderSource(FrameSource):
    """FrameSource streaming images from a folder sequentially."""

    def __init__(self, cfg: ImageFolderConfig) -> None:
        self._cfg = cfg
        self._paths: list[Path] = []
        self._idx = 0
        self._frame_id = 0

    def open(self) -> None:
        self._paths = list_image_files(self._cfg.directory, recursive=self._cfg.recursive)
        if not self._paths:
            raise ValueError(f"No images found in: {self._cfg.directory}")

        self._idx = 0
        self._frame_id = 0
        LOGGER.info("ImageFolder opened: %s (count=%d)", self._cfg.directory, len(self._paths))

    def read(self) -> Tuple[Optional[np.ndarray], Optional[FrameMeta]]:
        if not self._paths:
            raise RuntimeError("ImageFolderSource.read() called before open().")

        if self._idx >= len(self._paths):
            if self._cfg.loop:
                self._idx = 0
            else:
                return None, None

        path = self._paths[self._idx]
        self._idx += 1

        frame = cv.imread(str(path), cv.IMREAD_COLOR)
        if frame is None:
            LOGGER.warning("Failed to decode image, skipping: %s", path)
            return self.read()

        meta = FrameMeta(
            frame_id=self._frame_id,
            timestamp_s=time.perf_counter(),
            source=f"images:{self._cfg.directory.name}/{path.name}",
        )
        self._frame_id += 1
        return frame, meta

    def release(self) -> None:
        self._paths = []
        LOGGER.info("ImageFolder released.")