from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np

from src.camera_input.base import FrameSource
from src.utils.io import list_image_files
from src.utils.types import FrameMeta

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageFileConfig:
    path: Path
    loop: bool = False


class ImageFileSource(FrameSource):
    """Return a single image once or in a loop."""

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
        LOGGER.info("Image opened: %s", self._cfg.path)

    def read(self) -> tuple[Optional[np.ndarray], Optional[FrameMeta]]:
        if self._frame is None:
            raise RuntimeError("ImageFileSource.read() called before open().")
        if self._done and not self._cfg.loop:
            return None, None
        meta = FrameMeta(frame_id=self._frame_id, timestamp_s=time.perf_counter(), source=f"image:{self._cfg.path.name}")
        self._frame_id += 1
        self._done = True
        return self._frame.copy(), meta

    def release(self) -> None:
        self._frame = None


@dataclass(frozen=True)
class ImageFolderConfig:
    directory: Path
    loop: bool = False
    recursive: bool = False


class ImageFolderSource(FrameSource):
    """Stream images from a directory in sorted order."""

    def __init__(self, cfg: ImageFolderConfig) -> None:
        self._cfg = cfg
        self._paths: list[Path] = []
        self._index = 0
        self._frame_id = 0

    def open(self) -> None:
        self._paths = list_image_files(self._cfg.directory, recursive=self._cfg.recursive)
        if not self._paths:
            raise ValueError(f"No images found in {self._cfg.directory}")
        self._index = 0
        self._frame_id = 0
        LOGGER.info("Image folder opened: %s (%d images)", self._cfg.directory, len(self._paths))

    def read(self) -> tuple[Optional[np.ndarray], Optional[FrameMeta]]:
        if not self._paths:
            raise RuntimeError("ImageFolderSource.read() called before open().")
        if self._index >= len(self._paths):
            if not self._cfg.loop:
                return None, None
            self._index = 0

        path = self._paths[self._index]
        self._index += 1
        frame = cv.imread(str(path), cv.IMREAD_COLOR)
        if frame is None:
            LOGGER.warning("Could not decode image: %s", path)
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
