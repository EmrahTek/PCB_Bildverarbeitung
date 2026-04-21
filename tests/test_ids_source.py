from __future__ import annotations

import ctypes

import numpy as np

from src.camera_input.ids import (
    IDSCameraConfig,
    IDSCameraSource,
    PyUeyeConfig,
    PyUeyeSource,
    discover_video_devices,
    is_ids_device_name,
)


def _video_name(tmp_path, index: int, name: str) -> None:
    video_dir = tmp_path / f"video{index}"
    video_dir.mkdir()
    (video_dir / "name").write_text(name, encoding="utf-8")


def test_ids_device_name_detection() -> None:
    assert is_ids_device_name("IDS UI-3250CP-M-GL")
    assert is_ids_device_name("uEye Camera")
    assert not is_ids_device_name("Integrated Webcam")


def test_discover_video_devices_reads_sysfs_names(tmp_path) -> None:
    _video_name(tmp_path, 0, "Integrated Webcam")
    _video_name(tmp_path, 2, "IDS UI-3250CP-M-GL")

    devices = discover_video_devices(tmp_path)

    assert [(device.index, device.path, device.name) for device in devices] == [
        (0, "/dev/video0", "Integrated Webcam"),
        (2, "/dev/video2", "IDS UI-3250CP-M-GL"),
    ]


def test_ids_auto_selects_ids_v4l2_device(tmp_path) -> None:
    _video_name(tmp_path, 0, "Integrated Webcam")
    _video_name(tmp_path, 2, "IDS UI-3250CP-M-GL")
    source = IDSCameraSource(IDSCameraConfig(backend="auto", video_sys_path=tmp_path))

    opencv_cfg = source._select_opencv_config()

    assert opencv_cfg is not None
    assert opencv_cfg.device == "/dev/video2"
    assert opencv_cfg.backend == "v4l2,any"
    assert opencv_cfg.source_name == "ids-opencv"


def test_ids_allows_explicit_non_ids_device_with_warning_policy(tmp_path) -> None:
    _video_name(tmp_path, 0, "Integrated Webcam")
    _video_name(tmp_path, 2, "IDS UI-3250CP-M-GL")
    source = IDSCameraSource(IDSCameraConfig(device="0", backend="v4l2", video_sys_path=tmp_path))

    opencv_cfg = source._select_opencv_config()

    assert opencv_cfg is not None
    assert opencv_cfg.device == "/dev/video0"
    assert opencv_cfg.backend == "v4l2"


def test_ids_allows_explicit_unverified_opencv_override(tmp_path) -> None:
    _video_name(tmp_path, 0, "Integrated Webcam")
    source = IDSCameraSource(
        IDSCameraConfig(
            device="0",
            backend="v4l2",
            allow_unverified_opencv=True,
            video_sys_path=tmp_path,
        )
    )

    opencv_cfg = source._select_opencv_config()

    assert opencv_cfg is not None
    assert opencv_cfg.device == "/dev/video0"
    assert opencv_cfg.backend == "v4l2"


def test_ids_explicit_device_uses_opencv_before_pyueye_even_if_unverified(tmp_path) -> None:
    _video_name(tmp_path, 0, "Integrated Webcam")
    source = IDSCameraSource(IDSCameraConfig(device="0", backend="auto", video_sys_path=tmp_path))

    opencv_cfg = source._select_opencv_config()

    assert opencv_cfg is not None
    assert opencv_cfg.device == "/dev/video0"
    assert opencv_cfg.backend == "v4l2,any"


def test_ids_without_explicit_device_opens_default_index_when_no_ids_name_found(tmp_path) -> None:
    _video_name(tmp_path, 0, "Integrated Webcam")
    source = IDSCameraSource(IDSCameraConfig(index=1, backend="auto", video_sys_path=tmp_path))

    opencv_cfg = source._select_opencv_config()

    assert opencv_cfg is not None
    assert opencv_cfg.device is None
    assert opencv_cfg.index == 1
    assert opencv_cfg.backend == "v4l2,any"


def test_pyueye_memory_setup_uses_ctypes_output_parameters() -> None:
    calls: dict[str, bool] = {}

    class FakeUeye:
        HIDS = ctypes.c_uint
        INT = ctypes.c_int
        int = ctypes.c_int
        c_mem_p = ctypes.c_void_p
        IS_SUCCESS = 0
        IS_DONT_WAIT = 0
        IS_FORCE_VIDEO_STOP = 0

        @staticmethod
        def is_AllocImageMem(_h_cam, width, height, bits, image_mem, mem_id):
            calls["alloc_types"] = all(isinstance(value, ctypes.c_int) for value in (width, height, bits, mem_id))
            assert isinstance(image_mem, ctypes.c_void_p)
            image_mem.value = 1234
            mem_id.value = 7
            return 0

        @staticmethod
        def is_SetImageMem(_h_cam, image_mem, mem_id):
            calls["set_mem"] = isinstance(image_mem, ctypes.c_void_p) and isinstance(mem_id, ctypes.c_int)
            return 0

        @staticmethod
        def is_InquireImageMem(_h_cam, _image_mem, _mem_id, width, height, bits, pitch):
            calls["inquire_types"] = all(isinstance(value, ctypes.c_int) for value in (width, height, bits, pitch))
            width.value = 1600
            height.value = 1200
            bits.value = 24
            pitch.value = 4800
            return 0

    source = PyUeyeSource(PyUeyeConfig())
    source._ueye = FakeUeye
    source._h_cam = FakeUeye.HIDS(0)
    source._width = 1600
    source._height = 1200

    source._setup_image_memory()

    assert calls == {"alloc_types": True, "set_mem": True, "inquire_types": True}
    assert source._width == 1600
    assert source._height == 1200
    assert source._bits_per_pixel == 24
    assert source._bytes_per_pixel == 3
    assert source._pitch.value == 4800


def test_pyueye_data_to_frame_handles_pitch_padding() -> None:
    source = PyUeyeSource(PyUeyeConfig())
    source._width = 2
    source._height = 2
    source._bits_per_pixel = 24
    source._bytes_per_pixel = 3
    source._pitch = ctypes.c_int(8)
    data = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            99,
            99,
            7,
            8,
            9,
            10,
            11,
            12,
            88,
            88,
        ],
        dtype=np.uint8,
    )

    frame = source._data_to_frame(data)

    assert frame.shape == (2, 2, 3)
    assert frame.tolist() == [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]
