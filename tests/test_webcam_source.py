from __future__ import annotations

import numpy as np

from src.camera_input import webcam as webcam_module
from src.camera_input.webcam import WebcamConfig, WebcamSource


def test_webcam_source_parses_backend_sequence() -> None:
    source = WebcamSource(WebcamConfig(backend="ueye,v4l2,any", source_name="ids"))
    sequence = source._backend_sequence()
    assert [name for name, _backend_id in sequence][-2:] == ["v4l2", "any"]


def test_webcam_source_treats_numeric_device_as_integer_index() -> None:
    source = WebcamSource(WebcamConfig(device="0", backend="v4l2"))
    target = source._resolve_capture_target()
    assert target.raw == "0"
    assert target.value == 0
    assert target.kind == "integer-index"


def test_webcam_source_treats_dev_video_path_as_integer_index() -> None:
    source = WebcamSource(WebcamConfig(device="/dev/video2", backend="v4l2"))
    target = source._resolve_capture_target()
    assert target.raw == "/dev/video2"
    assert target.value == 2
    assert target.kind == "dev-video-index"


def test_webcam_source_skips_unavailable_explicit_backend(monkeypatch) -> None:
    monkeypatch.setattr(WebcamSource, "_backend_id", staticmethod(lambda name: {"ueye": 2500, "v4l2": 200, "any": 0}[name]))
    monkeypatch.setattr(WebcamSource, "_backend_available", staticmethod(lambda name, _backend_id: name != "ueye"))

    source = WebcamSource(WebcamConfig(backend="ueye,v4l2,any", source_name="ids"))
    sequence = source._backend_sequence()

    assert [name for name, _backend_id in sequence] == ["v4l2", "any"]


def test_webcam_open_passes_numeric_device_to_opencv(monkeypatch) -> None:
    calls: list[tuple[int | str, int]] = []

    class FakeCapture:
        def __init__(self, target: int | str, backend: int) -> None:
            calls.append((target, backend))
            self._opened = isinstance(target, int) and target == 0

        def isOpened(self) -> bool:
            return self._opened

        def release(self) -> None:
            self._opened = False

        def set(self, _prop: int, _value: float) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == webcam_module.cv.CAP_PROP_FRAME_WIDTH:
                return 640.0
            if prop == webcam_module.cv.CAP_PROP_FRAME_HEIGHT:
                return 480.0
            if prop == webcam_module.cv.CAP_PROP_FPS:
                return 30.0
            return 0.0

        def getBackendName(self) -> str:
            return "FAKE"

        def read(self) -> tuple[bool, np.ndarray]:
            return True, np.zeros((4, 6, 3), dtype=np.uint8)

    monkeypatch.setattr(webcam_module.cv, "VideoCapture", FakeCapture)
    source = WebcamSource(WebcamConfig(device="0", backend="any", use_mjpg=False))

    source.open()
    frame, meta = source.read()
    source.release()

    assert calls[0][0] == 0
    assert frame is not None and frame.shape == (4, 6, 3)
    assert meta is not None and meta.source == "webcam:0"
