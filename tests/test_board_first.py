from __future__ import annotations

import numpy as np

from src.detection_logic.board_first import BoardFirstConfig, BoardFirstDetector, ComponentSpec, RelativeROI
from src.detection_logic.template_match import TemplateMatchResult
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


class StatsMatcher:
    def __init__(self, result: TemplateMatchResult) -> None:
        self._result = result

    def detect_best_with_stats(self, frame: np.ndarray) -> TemplateMatchResult:
        return self._result


def _identity_localization(
    *,
    score: float = 0.90,
    warp_quality_score: float = 1.0,
    warped: np.ndarray | None = None,
) -> BoardLocalization:
    if warped is None:
        warped = np.zeros((100, 200, 3), dtype=np.uint8)
    homography = np.eye(3, dtype=np.float32)
    return BoardLocalization(
        quad=np.array([[0, 0], [199, 0], [199, 99], [0, 99]], dtype=np.float32),
        bbox=BBox(0, 0, 200, 100),
        homography=homography,
        h_inv=homography.copy(),
        warped=warped,
        score=score,
        warp_quality_score=warp_quality_score,
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


def test_board_first_suppresses_component_on_low_warp_quality() -> None:
    detector = BoardFirstDetector(
        localizer=DummyLocalizer(_identity_localization(warp_quality_score=0.25)),
        component_matchers={
            "ESP32": DummyMatcher(Detection(label="ESP32", score=0.95, bbox=BBox(20, 20, 120, 80))),
        },
        component_specs=[
            ComponentSpec(
                label="ESP32",
                roi=RelativeROI(0.0, 0.0, 1.0, 1.0),
                score_threshold=0.40,
                min_board_area_ratio=0.01,
                max_board_area_ratio=0.80,
                min_warp_quality_score=0.50,
            )
        ],
        cfg=BoardFirstConfig(temporal_window=1, temporal_min_hits=1),
    )

    detections = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))
    assert [det.label for det in detections] == ["BOARD"]


def test_layout_fallback_can_require_roi_match_evidence() -> None:
    detector = BoardFirstDetector(
        localizer=DummyLocalizer(_identity_localization()),
        component_matchers={
            "RESET_BUTTON": DummyMatcher(None),
        },
        component_specs=[
            ComponentSpec(
                label="RESET_BUTTON",
                roi=RelativeROI(0.0, 0.0, 1.0, 1.0),
                score_threshold=0.80,
                layout_roi=RelativeROI(0.45, 0.25, 0.58, 0.42),
                layout_fallback_score=0.55,
                layout_fallback_min_board_score=0.50,
                layout_fallback_min_warp_quality=0.50,
                layout_fallback_min_match_score=0.40,
                min_board_area_ratio=0.003,
                max_board_area_ratio=0.10,
                min_normalized_aspect_ratio=1.0,
                max_normalized_aspect_ratio=3.0,
            )
        ],
        cfg=BoardFirstConfig(temporal_window=1, temporal_min_hits=1),
    )

    detections = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))
    assert [det.label for det in detections] == ["BOARD"]


def test_component_visibility_can_rescue_near_threshold_match() -> None:
    candidate = Detection(label="ESP32", score=0.46, bbox=BBox(30, 15, 150, 80))
    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    crop[15:85, 30:160] = 210
    detector = BoardFirstDetector(
        localizer=DummyLocalizer(_identity_localization(warped=crop)),
        component_matchers={
            "ESP32": StatsMatcher(
                TemplateMatchResult(
                    detection=None,
                    best_score=0.46,
                    second_score=0.15,
                    score_margin=0.31,
                    reason="low_score",
                    candidate=candidate,
                )
            ),
        },
        component_specs=[
            ComponentSpec(
                label="ESP32",
                roi=RelativeROI(0.0, 0.0, 1.0, 1.0),
                score_threshold=0.50,
                min_board_area_ratio=0.01,
                max_board_area_ratio=0.80,
                min_visibility_score=0.10,
                visibility_weight=0.35,
                warp_quality_weight=0.08,
            )
        ],
        cfg=BoardFirstConfig(temporal_window=1, temporal_min_hits=1),
    )

    detections = detector.detect(crop)
    assert sorted(det.label for det in detections) == ["BOARD", "ESP32"]
