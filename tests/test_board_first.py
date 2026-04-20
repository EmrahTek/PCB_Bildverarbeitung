from __future__ import annotations

import numpy as np

from src.detection_logic.board_first import BoardFirstConfig, BoardFirstDetector, ComponentSpec, RelativeROI
from src.utils.types import BBox, Detection
from src.preprocessing.geometry import BoardLocalization


class DummyLocalizer:
    def __init__(self, localization: BoardLocalization) -> None:
        self._localization = localization

    def localize(
        self,
        frame: np.ndarray,
        hint_bbox: BBox | None = None,
        *,
        include_full_frame: bool = True,
    ) -> BoardLocalization | None:
        return self._localization


class DummyMatcher:
    def __init__(self, detection: Detection | None) -> None:
        self._detection = detection

    def detect_best(self, frame: np.ndarray) -> Detection | None:
        return self._detection


def _identity_localization() -> BoardLocalization:
    warped = np.zeros((100, 200, 3), dtype=np.uint8)
    homography = np.eye(3, dtype=np.float32)
    return BoardLocalization(
        quad=np.array([[0, 0], [199, 0], [199, 99], [0, 99]], dtype=np.float32),
        bbox=BBox(0, 0, 200, 100),
        homography=homography,
        h_inv=homography.copy(),
        warped=warped,
        score=0.90,
    )


def test_board_first_filters_component_that_is_too_small_for_board() -> None:
    detector = BoardFirstDetector(
        localizer=DummyLocalizer(_identity_localization()),
        component_matchers={
            "USB_PORT": DummyMatcher(Detection(label="USB_PORT", score=0.95, bbox=BBox(2, 2, 6, 6))),
        },
        component_specs=[
            ComponentSpec(
                label="USB_PORT",
                roi=RelativeROI(0.0, 0.0, 1.0, 1.0),
                score_threshold=0.40,
                min_board_area_ratio=0.01,
                max_board_area_ratio=0.10,
            )
        ],
        cfg=BoardFirstConfig(temporal_window=1, temporal_min_hits=1),
    )

    detections = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))
    assert [det.label for det in detections] == ["BOARD"]


def test_board_first_keeps_component_with_plausible_board_relative_size() -> None:
    detector = BoardFirstDetector(
        localizer=DummyLocalizer(_identity_localization()),
        component_matchers={
            "RESET_BUTTON": DummyMatcher(Detection(label="RESET_BUTTON", score=0.88, bbox=BBox(70, 35, 100, 55))),
        },
        component_specs=[
            ComponentSpec(
                label="RESET_BUTTON",
                roi=RelativeROI(0.0, 0.0, 1.0, 1.0),
                score_threshold=0.40,
                min_board_area_ratio=0.01,
                max_board_area_ratio=0.10,
                min_normalized_aspect_ratio=1.0,
                max_normalized_aspect_ratio=2.5,
            )
        ],
        cfg=BoardFirstConfig(temporal_window=1, temporal_min_hits=1),
    )

    detections = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))
    assert sorted(det.label for det in detections) == ["BOARD", "RESET_BUTTON"]


def test_board_first_uses_layout_fallback_when_template_score_is_missing() -> None:
    detector = BoardFirstDetector(
        localizer=DummyLocalizer(_identity_localization()),
        component_matchers={
            "USB_PORT": DummyMatcher(None),
        },
        component_specs=[
            ComponentSpec(
                label="USB_PORT",
                roi=RelativeROI(0.0, 0.0, 1.0, 1.0),
                score_threshold=0.80,
                layout_roi=RelativeROI(0.70, 0.20, 0.82, 0.40),
                layout_fallback_score=0.60,
                layout_fallback_min_board_score=0.50,
                min_board_area_ratio=0.005,
                max_board_area_ratio=0.10,
                min_normalized_aspect_ratio=1.0,
                max_normalized_aspect_ratio=3.0,
            )
        ],
        cfg=BoardFirstConfig(temporal_window=1, temporal_min_hits=1),
    )

    detections = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))
    labels = sorted(det.label for det in detections)
    assert labels == ["BOARD", "USB_PORT"]
    usb = next(det for det in detections if det.label == "USB_PORT")
    assert usb.score == 0.60
