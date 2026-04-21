from __future__ import annotations

import cv2 as cv
import numpy as np

from src.preprocessing.geometry import BoardLocalizer, BoardWarpConfig
from src.utils.types import BBox


def _synthetic_board_frame() -> np.ndarray:
    frame = np.full((420, 640, 3), 235, dtype=np.uint8)
    quad = np.array([[140, 120], [500, 100], [520, 280], [160, 300]], dtype=np.int32)
    cv.fillConvexPoly(frame, quad, (30, 30, 30))
    cv.rectangle(frame, (205, 150), (315, 255), (200, 200, 200), thickness=-1)
    cv.rectangle(frame, (430, 135), (485, 175), (210, 210, 210), thickness=-1)
    return frame


def _synthetic_portrait_board_frame() -> np.ndarray:
    frame = np.full((640, 420, 3), 235, dtype=np.uint8)
    quad = np.array([[140, 90], [300, 115], [260, 540], [100, 515]], dtype=np.int32)
    cv.fillConvexPoly(frame, quad, (30, 30, 30))
    cv.rectangle(frame, (150, 170), (245, 300), (200, 200, 200), thickness=-1)
    cv.rectangle(frame, (135, 420), (190, 485), (210, 210, 210), thickness=-1)
    return frame


def test_board_localizer_finds_prominent_quad() -> None:
    frame = _synthetic_board_frame()
    localizer = BoardLocalizer(BoardWarpConfig(output_size=(300, 150)))
    result = localizer.localize(frame)
    assert result is not None
    assert isinstance(result.bbox, BBox)
    assert result.bbox.area() > 30000
    assert result.warped.shape[:2] == (150, 300)


def test_board_localizer_normalizes_portrait_board_to_landscape_warp() -> None:
    frame = _synthetic_portrait_board_frame()
    localizer = BoardLocalizer(BoardWarpConfig(output_size=(300, 150)))
    result = localizer.localize(frame)
    assert result is not None
    assert isinstance(result.bbox, BBox)
    assert result.warped.shape[:2] == (150, 300)
    assert result.bbox.height() > result.bbox.width()
