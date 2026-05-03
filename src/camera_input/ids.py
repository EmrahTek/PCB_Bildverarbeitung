from __future__ import annotations

import importlib.util
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from src.camera_input.base import FrameSource
from src.camera_input.webcam import WebcamConfig, WebcamSource
from src.utils.types import FrameMeta

LOGGER = logging.getLogger(__name__)

IDS_NAME_PATTERNS = ("ids", "ueye", "u-eye", "ui-", "ui_", "ui3250", "ui-3250", "3250cp")


@dataclass(frozen=True)
class VideoDeviceInfo:
    index: int
    path: str
    name: str


@dataclass(frozen=True)
class IDSCameraConfig:
    """Configuration for IDS live acquisition."""

    index: int = 0
    device: str | None = None
    width: int | None = None
    height: int | None = None
    target_fps: int | None = None
    backend: str = "auto"
    use_mjpg: bool = False
    buffer_size: int = 1
    allow_unverified_opencv: bool = False
    video_sys_path: Path = Path("/sys/class/video4linux")


class IDSCameraSource(FrameSource):
    """
    IDS-aware frame source.

    The chosen acquisition path returns normal BGR numpy frames so the existing
    detector pipeline remains unchanged. Vendor identity checks are diagnostics:
    an explicitly selected OpenCV target is opened even when its Linux name is
    generic, because practical live feedback is more important than strict gating.
    """

    def __init__(self, cfg: IDSCameraConfig) -> None:
        self._cfg = cfg
        self._source: FrameSource | None = None

    def open(self) -> None:
        backend = self._cfg.backend.strip().lower()
        LOGGER.info(
            "IDS source open request: backend=%s raw_device=%s camera_index=%d width=%s height=%s fps=%s",
            backend,
            self._cfg.device,
            self._cfg.index,
            self._cfg.width,
            self._cfg.height,
            self._cfg.target_fps,
        )

        if backend == "pyueye":
            self._source = PyUeyeSource(self._pyueye_config())
        elif backend in {"ueye", "opencv-ueye"}:
            self._source = self._opencv_ueye_source()
        else:
            opencv_cfg = self._select_opencv_config()
            if opencv_cfg is not None:
                self._source = WebcamSource(opencv_cfg)
            elif backend == "auto" and pyueye_available():
                LOGGER.info("IDS V4L2 device not found; falling back to optional pyueye backend.")
                self._source = PyUeyeSource(self._pyueye_config())
            else:
                raise RuntimeError(self._no_ids_camera_message())

        self._source.open()

    def read(self) -> tuple[np.ndarray | None, FrameMeta | None]:
        if self._source is None:
            raise RuntimeError("IDSCameraSource.read() called before open().")
        return self._source.read()

    def release(self) -> None:
        if self._source is not None:
            self._source.release()
            self._source = None

    def _select_opencv_config(self) -> WebcamConfig | None:
        backend = self._cfg.backend.strip().lower()
        if backend in {"pyueye", "ueye", "opencv-ueye"}:
            return None

        opencv_backend = "v4l2,any" if backend in {"", "auto", "opencv"} else backend
        if "gstreamer" in opencv_backend:
            if not self._cfg.device:
                raise RuntimeError("--camera-device must contain a GStreamer pipeline when --camera-backend gstreamer.")
            LOGGER.warning(
                "IDS GStreamer target cannot be verified through /sys; using explicit pipeline as requested."
            )
            return self._opencv_config(device=self._cfg.device, backend=opencv_backend)

        discovered = discover_video_devices(self._cfg.video_sys_path)
        ids_devices = [device for device in discovered if is_ids_device_name(device.name)]
        explicit = self._explicit_video_device(discovered)

        if self._cfg.device is not None:
            if explicit is not None:
                verified = is_ids_device_name(explicit.name)
                self._log_opencv_selection(explicit, opencv_backend, verified=verified)
                if not verified:
                    LOGGER.warning(
                        "IDS OpenCV target is unverified but user-selected; opening anyway: "
                        "path=%s name='%s'. Detected IDS-looking devices: %s",
                        explicit.path,
                        explicit.name,
                        format_video_devices(ids_devices),
                    )
                return self._opencv_config(device=explicit.path, backend=opencv_backend)

            LOGGER.warning(
                "IDS OpenCV target has no sysfs identity but user-selected; opening anyway: "
                "raw_device=%s detected_video_devices=%s",
                self._cfg.device,
                format_video_devices(discovered),
            )
            return self._opencv_config(device=self._cfg.device, backend=opencv_backend)

        if ids_devices:
            selected = ids_devices[0]
            self._log_opencv_selection(selected, opencv_backend, verified=True)
            return self._opencv_config(device=selected.path, backend=opencv_backend)

        LOGGER.warning(
            "No IDS-looking V4L2 device found; opening camera index %d as unverified IDS OpenCV target. "
            "Detected video devices: %s",
            self._cfg.index,
            format_video_devices(discovered),
        )
        return self._opencv_config(device=None, backend=opencv_backend)

    def _opencv_ueye_source(self) -> WebcamSource:
        backend_id = getattr(cv, "CAP_UEYE", None)
        has_backend = False
        if backend_id is not None and hasattr(cv, "videoio_registry"):
            try:
                has_backend = bool(cv.videoio_registry.hasBackend(backend_id))
            except cv.error:
                has_backend = False
        if not has_backend:
            raise RuntimeError(
                "OpenCV CAP_UEYE is not available in this runtime. Use --camera-backend v4l2 "
                "with the IDS /dev/videoX device, or install pyueye and use --camera-backend pyueye."
            )
        LOGGER.info("IDS using OpenCV CAP_UEYE backend.")
        return WebcamSource(self._opencv_config(device=self._cfg.device, backend="ueye"))

    def _opencv_config(self, *, device: str | None, backend: str) -> WebcamConfig:
        return WebcamConfig(
            index=self._cfg.index,
            device=device,
            width=self._cfg.width,
            height=self._cfg.height,
            target_fps=self._cfg.target_fps,
            backend=backend,
            use_mjpg=self._cfg.use_mjpg,
            buffer_size=self._cfg.buffer_size,
            source_name="ids-opencv",
        )

    def _pyueye_config(self) -> "PyUeyeConfig":
        camera_id = self._cfg.index
        if self._cfg.device is not None and re.fullmatch(r"[+-]?\d+", self._cfg.device.strip()):
            camera_id = int(self._cfg.device.strip())
        return PyUeyeConfig(
            camera_id=camera_id,
            width=self._cfg.width,
            height=self._cfg.height,
            target_fps=self._cfg.target_fps,
        )

    def _explicit_video_device(self, discovered: list[VideoDeviceInfo]) -> VideoDeviceInfo | None:
        if self._cfg.device is None:
            return None
        raw = self._cfg.device.strip()
        index = _video_index_from_target(raw)
        if index is None:
            return None
        for device in discovered:
            if device.index == index:
                return device
        return None

    @staticmethod
    def _log_opencv_selection(device: VideoDeviceInfo, backend: str, *, verified: bool) -> None:
        LOGGER.info(
            "IDS using OpenCV/V4L2 path: path=%s index=%d name='%s' verified_ids=%s backend=%s",
            device.path,
            device.index,
            device.name,
            verified,
            backend,
        )

    def _no_ids_camera_message(self) -> str:
        devices = discover_video_devices(self._cfg.video_sys_path)
        return (
            "No IDS acquisition path is available. "
            f"Detected V4L2 devices: {format_video_devices(devices)}. "
            f"pyueye_installed={pyueye_available()}. "
            "Use --camera-device /dev/videoX or --camera-device N for the desired OpenCV/V4L2 target, "
            "or install pyueye and use --camera-backend pyueye."
        )


@dataclass(frozen=True)
class PyUeyeConfig:
    camera_id: int = 0
    width: int | None = None
    height: int | None = None
    target_fps: int | None = None


class PyUeyeSource(FrameSource):
    """Minimal optional IDS/uEye SDK source using pyueye."""

    def __init__(self, cfg: PyUeyeConfig) -> None:
        self._cfg = cfg
        self._ueye: Any | None = None
        self._h_cam: Any | None = None
        self._image_mem: Any | None = None
        self._mem_id: Any | None = None
        self._width = 0
        self._height = 0
        self._bits_per_pixel = 24
        self._bytes_per_pixel = 3
        self._pitch: Any | None = None
        self._frame_id = 0
        self._logged_first_frame = False

    def open(self) -> None:
        try:
            from pyueye import ueye  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "pyueye is not installed. Install the IDS Software Suite/uEye SDK and pyueye, "
                "or use the IDS V4L2 device with --camera-backend v4l2."
            ) from exc

        self._ueye = ueye
        self._h_cam = ueye.HIDS(int(self._cfg.camera_id))
        self._log_camera_count()
        self._check(ueye.is_InitCamera(self._h_cam, None), "is_InitCamera")
        LOGGER.info("pyueye camera initialized: camera_id=%d", self._cfg.camera_id)
        self._check(ueye.is_SetDisplayMode(self._h_cam, ueye.IS_SET_DM_DIB), "is_SetDisplayMode")
        self._check(ueye.is_SetColorMode(self._h_cam, ueye.IS_CM_BGR8_PACKED), "is_SetColorMode")
        LOGGER.info("pyueye color mode set: IS_CM_BGR8_PACKED bits=%d", self._bits_per_pixel)

        if self._cfg.target_fps is not None:
            new_fps = ueye.DOUBLE()
            self._check(
                ueye.is_SetFrameRate(self._h_cam, ueye.DOUBLE(float(self._cfg.target_fps)), new_fps),
                "is_SetFrameRate",
            )

        self._width, self._height = self._resolve_aoi()
        LOGGER.info("pyueye selected AOI: %dx%d", self._width, self._height)
        try:
            self._setup_image_memory()
            self._check(ueye.is_CaptureVideo(self._h_cam, ueye.IS_DONT_WAIT), "is_CaptureVideo")
        except RuntimeError as exc:
            self.release()
            raise RuntimeError(f"pyueye memory/capture setup failed after camera init: {exc}") from exc

        LOGGER.info(
            "IDS using pyueye path: camera_id=%d size=%dx%d fps_request=%s bits=%d bytes=%d pitch=%s",
            self._cfg.camera_id,
            self._width,
            self._height,
            self._cfg.target_fps,
            self._bits_per_pixel,
            self._bytes_per_pixel,
            getattr(self._pitch, "value", self._pitch),
        )

    def read(self) -> tuple[np.ndarray | None, FrameMeta | None]:
        if self._ueye is None or self._h_cam is None or self._image_mem is None or self._mem_id is None:
            raise RuntimeError("PyUeyeSource.read() called before open().")
        if self._pitch is None:
            raise RuntimeError("PyUeyeSource.read() called before image memory was configured.")

        try:
            data = self._ueye.get_data(
                self._image_mem,
                self._width,
                self._height,
                self._bits_per_pixel,
                self._pitch,
                copy=True,
            )
            frame = self._data_to_frame(data)
        except Exception as exc:
            LOGGER.warning("pyueye frame acquisition failed: %s", exc)
            return None, None

        if not self._logged_first_frame:
            LOGGER.info("pyueye first frame acquired: shape=%s dtype=%s", frame.shape, frame.dtype)
            self._logged_first_frame = True
        meta = FrameMeta(
            frame_id=self._frame_id,
            timestamp_s=time.perf_counter(),
            source=f"ids-pyueye:{self._cfg.camera_id}",
        )
        self._frame_id += 1
        return frame, meta

    def release(self) -> None:
        if self._ueye is None or self._h_cam is None:
            return
        if hasattr(self._ueye, "is_StopLiveVideo"):
            self._ueye.is_StopLiveVideo(self._h_cam, getattr(self._ueye, "IS_FORCE_VIDEO_STOP", 0))
        if self._image_mem is not None and self._mem_id is not None:
            self._ueye.is_FreeImageMem(self._h_cam, self._image_mem, self._mem_id)
        self._ueye.is_ExitCamera(self._h_cam)
        self._ueye = None
        self._h_cam = None
        self._image_mem = None
        self._mem_id = None
        self._pitch = None

    def _resolve_aoi(self) -> tuple[int, int]:
        assert self._ueye is not None
        assert self._h_cam is not None
        ueye = self._ueye
        rect = ueye.IS_RECT()
        self._check(ueye.is_AOI(self._h_cam, ueye.IS_AOI_IMAGE_GET_AOI, rect, ueye.sizeof(rect)), "is_AOI(GET)")
        width = int(rect.s32Width.value)
        height = int(rect.s32Height.value)

        if self._cfg.width is not None and self._cfg.height is not None:
            rect.s32X = ueye.int(0)
            rect.s32Y = ueye.int(0)
            rect.s32Width = ueye.int(int(self._cfg.width))
            rect.s32Height = ueye.int(int(self._cfg.height))
            result = ueye.is_AOI(self._h_cam, ueye.IS_AOI_IMAGE_SET_AOI, rect, ueye.sizeof(rect))
            if result == ueye.IS_SUCCESS:
                width = int(self._cfg.width)
                height = int(self._cfg.height)
            else:
                LOGGER.warning(
                    "pyueye AOI set failed for requested size %sx%s: code=%s; using sensor AOI %dx%d",
                    self._cfg.width,
                    self._cfg.height,
                    result,
                    width,
                    height,
                )
        return width, height

    def _setup_image_memory(self) -> None:
        assert self._ueye is not None
        assert self._h_cam is not None
        ueye = self._ueye

        width = ueye.INT(int(self._width))
        height = ueye.INT(int(self._height))
        bits = ueye.INT(int(self._bits_per_pixel))
        self._image_mem = ueye.c_mem_p()
        self._mem_id = ueye.int()

        self._check(
            ueye.is_AllocImageMem(
                self._h_cam,
                width,
                height,
                bits,
                self._image_mem,
                self._mem_id,
            ),
            f"is_AllocImageMem(width={self._as_int(width)}, height={self._as_int(height)}, bits={self._as_int(bits)})",
        )
        LOGGER.info(
            "pyueye image memory allocated: requested=%dx%d bits=%d",
            self._as_int(width),
            self._as_int(height),
            self._as_int(bits),
        )

        self._check(ueye.is_SetImageMem(self._h_cam, self._image_mem, self._mem_id), "is_SetImageMem")
        LOGGER.info("pyueye image memory set: mem_id=%s", getattr(self._mem_id, "value", self._mem_id))

        inquire_width = ueye.INT()
        inquire_height = ueye.INT()
        inquire_bits = ueye.INT()
        self._pitch = ueye.INT()
        self._check(
            ueye.is_InquireImageMem(
                self._h_cam,
                self._image_mem,
                self._mem_id,
                inquire_width,
                inquire_height,
                inquire_bits,
                self._pitch,
            ),
            "is_InquireImageMem",
        )

        self._width = max(1, self._as_int(inquire_width))
        self._height = max(1, self._as_int(inquire_height))
        self._bits_per_pixel = max(8, self._as_int(inquire_bits))
        self._bytes_per_pixel = max(1, self._bits_per_pixel // 8)
        LOGGER.info(
            "pyueye image memory inquiry OK: width=%d height=%d bits=%d pitch=%d",
            self._width,
            self._height,
            self._bits_per_pixel,
            self._as_int(self._pitch),
        )

    def _data_to_frame(self, data: np.ndarray) -> np.ndarray:
        if self._pitch is None:
            raise RuntimeError("pyueye pitch is not initialized.")
        pitch = self._as_int(self._pitch)
        row_bytes = int(self._width) * int(self._bytes_per_pixel)
        if pitch < row_bytes:
            raise RuntimeError(f"pyueye pitch {pitch} is smaller than expected row bytes {row_bytes}.")

        flat = np.asarray(data, dtype=np.uint8)
        expected = int(self._height) * pitch
        if flat.size < expected:
            raise RuntimeError(f"pyueye returned {flat.size} bytes, expected at least {expected}.")

        rows = flat[:expected].reshape((int(self._height), pitch))
        packed = rows[:, :row_bytes]
        frame = packed.reshape((int(self._height), int(self._width), int(self._bytes_per_pixel)))
        if frame.shape[2] > 3:
            frame = frame[:, :, :3]
        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        return np.ascontiguousarray(frame)

    def _check(self, result: int, operation: str) -> None:
        assert self._ueye is not None
        code = self._as_int(result)
        if code != self._as_int(self._ueye.IS_SUCCESS):
            raise RuntimeError(f"pyueye {operation} failed with code {code} ({self._error_name(code)})")

    def _log_camera_count(self) -> None:
        assert self._ueye is not None
        if not hasattr(self._ueye, "is_GetNumberOfCameras"):
            return
        try:
            count = self._ueye.INT()
            result = self._ueye.is_GetNumberOfCameras(count)
            if self._as_int(result) == self._as_int(self._ueye.IS_SUCCESS):
                LOGGER.info("pyueye reported connected camera count: %d", self._as_int(count))
        except Exception as exc:
            LOGGER.debug("pyueye camera count query failed: %s", exc)

    def _error_name(self, code: int) -> str:
        assert self._ueye is not None
        matches: list[str] = []
        for name in dir(self._ueye):
            if not name.startswith("IS_"):
                continue
            try:
                if self._as_int(getattr(self._ueye, name)) == code:
                    matches.append(name)
            except Exception:
                continue
        return "|".join(sorted(matches)) if matches else "unknown"

    @staticmethod
    def _as_int(value: Any) -> int:
        if hasattr(value, "value"):
            return int(value.value)
        return int(value)


def pyueye_available() -> bool:
    return importlib.util.find_spec("pyueye") is not None


def discover_video_devices(sys_class: Path = Path("/sys/class/video4linux")) -> list[VideoDeviceInfo]:
    devices: list[VideoDeviceInfo] = []
    if not sys_class.exists():
        return devices
    for entry in sorted(sys_class.glob("video*")):
        match = re.fullmatch(r"video(\d+)", entry.name)
        if match is None:
            continue
        name_path = entry / "name"
        try:
            name = name_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            name = ""
        index = int(match.group(1))
        devices.append(VideoDeviceInfo(index=index, path=f"/dev/video{index}", name=name))
    return devices


def is_ids_device_name(name: str) -> bool:
    lowered = name.strip().lower()
    return any(pattern in lowered for pattern in IDS_NAME_PATTERNS)


def format_video_devices(devices: list[VideoDeviceInfo]) -> str:
    if not devices:
        return "none"
    return ", ".join(f"{device.path}='{device.name or 'unknown'}'" for device in devices)


def _video_index_from_target(raw: str) -> int | None:
    raw = raw.strip()
    if re.fullmatch(r"[+-]?\d+", raw):
        return int(raw)
    dev_video = re.fullmatch(r"/dev/video(\d+)", raw)
    if dev_video is not None:
        return int(dev_video.group(1))
    return None
