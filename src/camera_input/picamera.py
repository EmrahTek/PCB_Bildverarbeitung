"""Raspberry Pi Picamera2 frame source.

This module imports Picamera2 lazily, configures the selected Pi camera, and
converts captured frames into OpenCV BGR arrays for the shared pipeline.

Python docs:
- dataclasses: https://docs.python.org/3/library/dataclasses.html
- importlib: https://docs.python.org/3/library/importlib.html
- sys: https://docs.python.org/3/library/sys.html
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

import cv2 as cv
import numpy as np

from src.camera_input.base import FrameSource
from src.utils.types import FrameMeta

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PiCameraConfig:
    """Configuration for Raspberry Pi libcamera/Picamera2 acquisition."""

    camera_num: int = 0
    width: int | None = None
    height: int | None = None
    target_fps: int | None = None
    buffer_count: int = 4
    pixel_format: str = "RGB888"


class PiCameraSource(FrameSource):
    """
    Raspberry Pi camera source backed by Picamera2.

    Picamera2 is imported lazily so the project still runs on non-Pi machines
    for image, video, webcam, IDS, and unit-test workflows.
    """

    def __init__(self, cfg: PiCameraConfig) -> None:
        self._cfg = cfg
        self._picam2: Any | None = None
        self._frame_id = 0

    def open(self) -> None:
        picamera2_module = _import_picamera2()
        picamera_cls = picamera2_module.Picamera2
        camera_info = _global_camera_info(picamera_cls)
        if camera_info is not None:
            if not camera_info:
                raise RuntimeError(
                    "Picamera2 did not report any cameras. Check the camera cable, "
                    "camera enablement, libcamera setup, and 'rpicam-hello --list-cameras'."
                )
            if int(self._cfg.camera_num) < 0 or int(self._cfg.camera_num) >= len(camera_info):
                raise RuntimeError(
                    f"Picamera2 camera index {self._cfg.camera_num} is out of range. "
                    f"Detected cameras: {_format_camera_info(camera_info)}"
                )

        try:
            self._picam2 = picamera_cls(camera_num=int(self._cfg.camera_num))
        except TypeError:
            try:
                self._picam2 = picamera_cls(int(self._cfg.camera_num))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to initialize Picamera2 camera {self._cfg.camera_num}. "
                    f"Detected cameras: {_format_camera_info(camera_info)}"
                ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize Picamera2 camera {self._cfg.camera_num}. "
                f"Detected cameras: {_format_camera_info(camera_info)}"
            ) from exc

        width, height = self._capture_size()
        controls: dict[str, float] = {}
        if self._cfg.target_fps is not None:
            controls["FrameRate"] = float(self._cfg.target_fps)

        kwargs: dict[str, Any] = {
            "main": {"size": (width, height), "format": self._cfg.pixel_format},
            "buffer_count": max(1, int(self._cfg.buffer_count)),
        }
        if controls:
            kwargs["controls"] = controls

        try:
            configuration = self._picam2.create_video_configuration(**kwargs)
            self._picam2.configure(configuration)
            self._picam2.start()
        except Exception:
            self.release()
            raise
        self._frame_id = 0
        LOGGER.info(
            "PiCamera opened: camera_num=%d size=%dx%d format=%s fps_request=%s buffer=%d",
            self._cfg.camera_num,
            width,
            height,
            self._cfg.pixel_format,
            self._cfg.target_fps,
            max(1, int(self._cfg.buffer_count)),
        )

    def read(self) -> tuple[np.ndarray | None, FrameMeta | None]:
        if self._picam2 is None:
            raise RuntimeError("PiCameraSource.read() called before open().")

        try:
            frame = self._picam2.capture_array("main")
        except Exception as exc:
            LOGGER.warning("Picamera2 frame acquisition failed: %s", exc)
            return None, None

        if frame is None:
            return None, None

        bgr_frame = self._to_bgr(np.asarray(frame))
        meta = FrameMeta(
            frame_id=self._frame_id,
            timestamp_s=time.perf_counter(),
            source=f"picamera:{self._cfg.camera_num}",
        )
        self._frame_id += 1
        return bgr_frame, meta

    def release(self) -> None:
        if self._picam2 is None:
            return
        try:
            self._picam2.stop()
        except Exception as exc:
            LOGGER.debug("Picamera2 stop failed during release: %s", exc)
        try:
            self._picam2.close()
        except Exception as exc:
            LOGGER.debug("Picamera2 close failed during release: %s", exc)
        finally:
            self._picam2 = None

    def _capture_size(self) -> tuple[int, int]:
        width = 1280 if self._cfg.width is None else int(self._cfg.width)
        height = 720 if self._cfg.height is None else int(self._cfg.height)
        if width <= 0 or height <= 0:
            raise ValueError("PiCamera width and height must be positive")
        return width, height

    def _to_bgr(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

        if frame.ndim != 3:
            raise RuntimeError(f"Unexpected Picamera2 frame shape: {frame.shape}")

        channels = frame.shape[2]
        pixel_format = self._cfg.pixel_format.upper()
        if channels == 3:
            if pixel_format.startswith("BGR"):
                return np.ascontiguousarray(frame)
            return cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        if channels == 4:
            if pixel_format.startswith(("BGR", "XBGR")):
                return cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
            return cv.cvtColor(frame, cv.COLOR_RGBA2BGR)

        raise RuntimeError(f"Unexpected Picamera2 channel count: {channels}")


def picamera2_available() -> bool:
    return importlib.util.find_spec("picamera2") is not None


def _import_picamera2() -> Any:
    try:
        return importlib.import_module("picamera2")
    except ImportError as exc:
        raise RuntimeError(
            "Picamera2 is not installed. On Raspberry Pi OS install it with "
            "'sudo apt install python3-picamera2', or run this project with a "
            "Python environment that can import the apt Picamera2 package. "
            f"Current Python is {sys.executable}. {_venv_hint()}"
        ) from exc


def _venv_hint() -> str:
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    if sys.prefix != base_prefix:
        return "A virtualenv is active; use 'deactivate' first or run '/usr/bin/python3 main.py ...'."
    return "If a virtualenv prompt is active, use '/usr/bin/python3 main.py ...' explicitly."


def _global_camera_info(picamera_cls: Any) -> list[Any] | None:
    global_camera_info = getattr(picamera_cls, "global_camera_info", None)
    if not callable(global_camera_info):
        return None
    try:
        info = global_camera_info()
    except Exception as exc:
        LOGGER.debug("Picamera2 global camera info query failed: %s", exc)
        return None
    if not isinstance(info, list):
        return None
    return info


def _format_camera_info(camera_info: list[Any] | None) -> str:
    if camera_info is None:
        return "unavailable"
    if not camera_info:
        return "none"

    formatted: list[str] = []
    for index, item in enumerate(camera_info):
        if isinstance(item, dict):
            model = item.get("Model") or item.get("Id") or item.get("Num") or "unknown"
            formatted.append(f"{index}:{model}")
        else:
            formatted.append(f"{index}:{item}")
    return ", ".join(formatted)
