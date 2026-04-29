from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.app.cli import parse_args
from src.camera_input import picamera as picamera_module
from src.camera_input.picamera import PiCameraConfig, PiCameraSource


def test_cli_accepts_picamera_source() -> None:
    args = parse_args(["--source", "picamera"])

    assert args.source == "picamera"


def test_picamera_source_uses_picamera2_and_returns_bgr(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakePicamera2:
        def __init__(self, camera_num: int = 0) -> None:
            calls["camera_num"] = camera_num

        def create_video_configuration(self, **kwargs):
            calls["configuration_kwargs"] = kwargs
            return {"configured": True}

        def configure(self, configuration) -> None:
            calls["configured"] = configuration

        def start(self) -> None:
            calls["started"] = True

        def capture_array(self, stream_name: str):
            calls["stream_name"] = stream_name
            return np.array([[[10, 20, 30]]], dtype=np.uint8)

        def stop(self) -> None:
            calls["stopped"] = True

        def close(self) -> None:
            calls["closed"] = True

    def fake_import_module(name: str):
        assert name == "picamera2"
        return SimpleNamespace(Picamera2=FakePicamera2)

    monkeypatch.setattr(picamera_module.importlib, "import_module", fake_import_module)

    source = PiCameraSource(
        PiCameraConfig(
            camera_num=1,
            width=640,
            height=480,
            target_fps=30,
            buffer_count=2,
        )
    )

    source.open()
    frame, meta = source.read()
    source.release()

    assert calls["camera_num"] == 1
    assert calls["configuration_kwargs"] == {
        "main": {"size": (640, 480), "format": "RGB888"},
        "buffer_count": 2,
        "controls": {"FrameRate": 30.0},
    }
    assert frame is not None
    assert frame.tolist() == [[[30, 20, 10]]]
    assert meta is not None
    assert meta.source == "picamera:1"
    assert calls["stopped"] is True
    assert calls["closed"] is True
