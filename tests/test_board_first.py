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


class SequenceLocalizer:
    def __init__(self, localizations: list[BoardLocalization]) -> None:
        self._localizations = list(localizations)
        self.calls = 0

    def localize(
        self,
        frame: np.ndarray,
        hint_bbox: BBox | None = None,
        *,
        include_full_frame: bool = True,
    ) -> BoardLocalization | None:
        index = min(self.calls, len(self._localizations) - 1)
        self.calls += 1
        return self._localizations[index]


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


class SequenceStatsMatcher:
    def __init__(self, results: list[TemplateMatchResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def detect_best_with_stats(self, frame: np.ndarray) -> TemplateMatchResult:
        index = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[index]


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


def test_board_bbox_is_not_averaged_by_generic_temporal_filter() -> None:
    first = _identity_localization()
    second = BoardLocalization(
        quad=np.array([[20, 0], [219, 0], [219, 99], [20, 99]], dtype=np.float32),
        bbox=BBox(20, 0, 220, 100),
        homography=np.eye(3, dtype=np.float32),
        h_inv=np.eye(3, dtype=np.float32),
        warped=np.zeros((100, 200, 3), dtype=np.uint8),
        score=0.90,
        warp_quality_score=1.0,
    )
    detector = BoardFirstDetector(
        localizer=SequenceLocalizer([first, second]),
        component_matchers={},
        component_specs=[],
        cfg=BoardFirstConfig(temporal_window=3, temporal_min_hits=1, enable_tracking=False),
    )

    detector.detect(np.zeros((100, 240, 3), dtype=np.uint8))
    detections = detector.detect(np.zeros((100, 240, 3), dtype=np.uint8))

    assert detections == [Detection(label="BOARD", score=0.90, bbox=BBox(20, 0, 220, 100))]


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


def test_component_lock_searches_near_previous_canonical_bbox() -> None:
    first = TemplateMatchResult(
        detection=Detection(label="USB_PORT", score=0.82, bbox=BBox(100, 40, 140, 70)),
        best_score=0.82,
        second_score=0.20,
        score_margin=0.62,
    )
    second_local = TemplateMatchResult(
        detection=Detection(label="USB_PORT", score=0.76, bbox=BBox(5, 5, 45, 35)),
        best_score=0.76,
        second_score=0.20,
        score_margin=0.56,
    )
    matcher = SequenceStatsMatcher([first, second_local])
    detector = BoardFirstDetector(
        localizer=DummyLocalizer(_identity_localization()),
        component_matchers={"USB_PORT": matcher},
        component_specs=[
            ComponentSpec(
                label="USB_PORT",
                roi=RelativeROI(0.0, 0.0, 1.0, 1.0),
                score_threshold=0.50,
                keep_score_threshold=0.45,
                min_board_area_ratio=0.001,
                max_board_area_ratio=0.50,
                local_search_expansion=0.50,
                track_max_missing=2,
            )
        ],
        cfg=BoardFirstConfig(temporal_window=1, temporal_min_hits=1, enable_tracking=True),
    )

    first_detections = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))
    second_detections = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert "USB_PORT" in [det.label for det in first_detections]
    usb = next(det for det in second_detections if det.label == "USB_PORT")
    assert usb.bbox.x1 > 70
    assert matcher.calls == 2


def test_reset_button_can_persist_from_local_track_evidence() -> None:
    warped = np.zeros((100, 200, 3), dtype=np.uint8)
    warped[35:55, 75:105] = 210
    first = TemplateMatchResult(
        detection=Detection(label="RESET_BUTTON", score=0.78, bbox=BBox(75, 35, 105, 55)),
        best_score=0.78,
        second_score=0.10,
        score_margin=0.68,
    )
    missing = TemplateMatchResult(
        detection=None,
        best_score=-1.0,
        second_score=-1.0,
        score_margin=1.0,
        reason="no_valid_template",
    )
    detector = BoardFirstDetector(
        localizer=DummyLocalizer(_identity_localization(warped=warped)),
        component_matchers={"RESET_BUTTON": SequenceStatsMatcher([first, missing])},
        component_specs=[
            ComponentSpec(
                label="RESET_BUTTON",
                roi=RelativeROI(0.0, 0.0, 1.0, 1.0),
                score_threshold=0.60,
                keep_score_threshold=0.50,
                keep_min_visibility_score=0.05,
                min_board_area_ratio=0.001,
                max_board_area_ratio=0.20,
                local_search_expansion=0.90,
                track_max_missing=3,
                persistence_decay=0.92,
            )
        ],
        cfg=BoardFirstConfig(temporal_window=1, temporal_min_hits=1, enable_tracking=True),
    )

    detector.detect(warped)
    detections = detector.detect(warped)

    assert "RESET_BUTTON" in [det.label for det in detections]


def test_active_component_track_defers_full_roi_search_until_lost() -> None:
    first = TemplateMatchResult(
        detection=Detection(label="USB_PORT", score=0.82, bbox=BBox(100, 40, 140, 70)),
        best_score=0.82,
        second_score=0.20,
        score_margin=0.62,
    )
    missing = TemplateMatchResult(
        detection=None,
        best_score=-1.0,
        second_score=-1.0,
        score_margin=1.0,
        reason="no_valid_template",
    )
    full_roi_decoy = TemplateMatchResult(
        detection=Detection(label="USB_PORT", score=0.99, bbox=BBox(0, 0, 40, 30)),
        best_score=0.99,
        second_score=0.20,
        score_margin=0.79,
    )
    matcher = SequenceStatsMatcher([first, missing, full_roi_decoy])
    detector = BoardFirstDetector(
        localizer=DummyLocalizer(_identity_localization()),
        component_matchers={"USB_PORT": matcher},
        component_specs=[
            ComponentSpec(
                label="USB_PORT",
                roi=RelativeROI(0.0, 0.0, 1.0, 1.0),
                score_threshold=0.50,
                keep_score_threshold=0.80,
                min_board_area_ratio=0.001,
                max_board_area_ratio=0.50,
                local_search_expansion=0.50,
                track_max_missing=2,
            )
        ],
        cfg=BoardFirstConfig(temporal_window=1, temporal_min_hits=1, enable_tracking=True),
    )

    detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))
    detections = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert matcher.calls == 2
    assert [det.label for det in detections] == ["BOARD"]
