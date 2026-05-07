"""OpenCV webcam frame source.

This module resolves Linux camera targets and OpenCV backend choices, opens the
selected webcam, and returns BGR frames with FrameMeta records.

Python docs:
- dataclasses: https://docs.python.org/3/library/dataclasses.html
- logging: https://docs.python.org/3/library/logging.html
- re: https://docs.python.org/3/library/re.html
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.camera_input.base import FrameSource
from src.utils.types import FrameMeta

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CaptureTarget:
    raw: str
    value: int | str
    kind: str


@dataclass(frozen=True)
class WebcamConfig:
    """Configuration for the OpenCV webcam source."""
    index: int = 0
    device: str | None = None
    width: int | None = None
    height: int | None = None
    target_fps: int | None = None
    backend: str = "auto"
    use_mjpg: bool = True
    buffer_size: int = 1
    source_name: str = "webcam"


class WebcamSource(FrameSource):
    """Live frame source backed by cv2.VideoCapture."""

    def __init__(self, cfg: WebcamConfig) -> None:
        self._cfg = cfg
        self._cap: cv.VideoCapture | None = None
        self._frame_id = 0

    def open(self) -> None:
        capture_target = self._resolve_capture_target()
        backend_sequence = self._backend_sequence()
        if not backend_sequence:
            raise RuntimeError(
                f"No usable camera backend for {self._cfg.source_name}. "
                f"Requested backend='{self._cfg.backend}'. {self._backend_help()}"
            )

        LOGGER.info(
            "%s capture open request: raw_target=%s interpreted=%s value=%r backend_request=%s",
            self._cfg.source_name.upper(),
            capture_target.raw,
            capture_target.kind,
            capture_target.value,
            self._cfg.backend,
        )

        attempted: list[str] = []
        for backend_name, backend_id in backend_sequence:
            attempted.append(backend_name)
            LOGGER.info(
                "%s trying backend=%s target_kind=%s target_value=%r",
                self._cfg.source_name.upper(),
                backend_name,
                capture_target.kind,
                capture_target.value,
            )
            self._cap = cv.VideoCapture(capture_target.value, backend_id)
            if self._cap.isOpened():
                LOGGER.info(
                    "%s capture opened with requested_backend=%s target_kind=%s target_value=%r",
                    self._cfg.source_name.upper(),
                    backend_name,
                    capture_target.kind,
                    capture_target.value,
                )
                break
            self._cap.release()
            self._cap = None
            LOGGER.debug(
                "%s backend failed: backend=%s target_kind=%s target_value=%r",
                self._cfg.source_name.upper(),
                backend_name,
                capture_target.kind,
                capture_target.value,
            )

        if self._cap is None:
            raise RuntimeError(
                f"Failed to open {self._cfg.source_name} target {capture_target.raw} "
                f"(interpreted as {capture_target.kind} {capture_target.value!r}) "
                f"with backends {attempted}. {self._backend_help()}"
            )
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open {self._cfg.source_name} target {capture_target.raw} "
                f"(interpreted as {capture_target.kind} {capture_target.value!r})"
            )

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
            "%s opened: raw_target=%s target_kind=%s target_value=%r size=%dx%d fps=%.1f "
            "mjpg=%s buffer=%d final_backend=%s",
            self._cfg.source_name.capitalize(),
            capture_target.raw,
            capture_target.kind,
            capture_target.value,
            actual_w,
            actual_h,
            actual_fps,
            self._cfg.use_mjpg,
            self._cfg.buffer_size,
            self._backend_name(),
        )

    def read(self) -> tuple[np.ndarray | None, FrameMeta | None]:
        if self._cap is None:
            raise RuntimeError("WebcamSource.read() called before open().")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None, None
        meta = FrameMeta(
            frame_id=self._frame_id,
            timestamp_s=time.perf_counter(),
            source=f"{self._cfg.source_name}:{self._resolve_capture_target().raw}",
        )
        self._frame_id += 1
        return frame, meta

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _backend_sequence(self) -> list[tuple[str, int]]:
        backend = self._cfg.backend.strip().lower()
        if backend in {"", "auto"}:
            return [("any", cv.CAP_ANY)]

        sequence: list[tuple[str, int]] = []
        for raw_name in backend.split(","):
            name = raw_name.strip().lower()
            if not name:
                continue
            backend_id = self._backend_id(name)
            if backend_id is None:
                LOGGER.warning("Camera backend '%s' is not known by this OpenCV build; skipping.", name)
                continue
            if not self._backend_available(name, backend_id):
                LOGGER.warning("Camera backend '%s' is not available in this OpenCV runtime; skipping.", name)
                continue
            sequence.append((name, backend_id))
        return sequence

    @staticmethod
    def _backend_id(name: str) -> int | None:
        mapping = {
            "any": cv.CAP_ANY,
            "auto": cv.CAP_ANY,
            "v4l2": getattr(cv, "CAP_V4L2", None),
            "ueye": getattr(cv, "CAP_UEYE", None),
            "gstreamer": getattr(cv, "CAP_GSTREAMER", None),
            "dshow": getattr(cv, "CAP_DSHOW", None),
            "msmf": getattr(cv, "CAP_MSMF", None),
            "avfoundation": getattr(cv, "CAP_AVFOUNDATION", None),
        }
        if name not in mapping:
            LOGGER.warning("Unknown camera backend '%s'; skipping.", name)
        return mapping.get(name)

    @staticmethod
    def _backend_available(name: str, backend_id: int) -> bool:
        if name in {"any", "auto"}:
            return True
        registry = getattr(cv, "videoio_registry", None)
        has_backend = getattr(registry, "hasBackend", None)
        if not callable(has_backend):
            return True
        try:
            return bool(has_backend(backend_id))
        except cv.error:
            return True

    def _resolve_capture_target(self) -> _CaptureTarget:
        if self._cfg.device is None:
            return _CaptureTarget(raw=str(self._cfg.index), value=int(self._cfg.index), kind="integer-index")

        raw = self._cfg.device.strip()
        if re.fullmatch(r"[+-]?\d+", raw):
            return _CaptureTarget(raw=raw, value=int(raw), kind="integer-index")

        dev_video = re.fullmatch(r"/dev/video(\d+)", raw)
        if dev_video is not None:
            return _CaptureTarget(raw=raw, value=int(dev_video.group(1)), kind="dev-video-index")

        if "!" in raw or raw.lower().startswith(("v4l2src ", "libcamerasrc ", "rtspsrc ", "filesrc ")):
            return _CaptureTarget(raw=raw, value=raw, kind="pipeline")

        return _CaptureTarget(raw=raw, value=raw, kind="path-or-name")

    def _backend_help(self) -> str:
        if self._cfg.source_name == "ids":
            return (
                "IDS is supported here through OpenCV backends only. If CAP_UEYE is unavailable, "
                "expose the camera as V4L2 and use --camera-backend v4l2 with --camera-index N "
                "or --camera-device /dev/videoN."
            )
        return "Try --camera-index N, --camera-device /dev/videoN, or --camera-backend any."

    def _backend_name(self) -> str:
        if self._cap is None:
            return "none"
        try:
            return self._cap.getBackendName()
        except cv.error:
            return self._cfg.backend
