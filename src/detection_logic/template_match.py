from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.detection_logic.base import Detector
from src.preprocessing.filters import MatchPrepConfig, prepare_match_images
from src.utils.types import BBox, Detection


@dataclass(frozen=True)
class TemplateMatchConfig:
    """Configuration for a component-specific template matcher."""
    label: str
    score_threshold: float
    scales: tuple[float, ...]
    gray_weight: float = 0.65
    edge_weight: float = 0.35
    use_clahe: bool = True
    blur_ksize: int = 3
    min_template_size: int = 12
    min_score_margin: float = 0.0
    second_best_iou_threshold: float = 0.45


@dataclass(frozen=True)
class _PreparedTemplate:
    gray: np.ndarray
    edges: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class TemplateMatchResult:
    """Best template hit plus ambiguity diagnostics."""
    detection: Detection | None
    best_score: float
    second_score: float
    score_margin: float
    reason: str = ""


class TemplateMatcher(Detector):
    """
    Multi-scale matcher that combines grayscale and edge correlation.

    The implementation intentionally returns only the best hit because every ROI in
    this project is expected to contain at most one component instance.
    """

    def __init__(self, templates_gray: list[np.ndarray], cfg: TemplateMatchConfig) -> None:
        if not templates_gray:
            raise ValueError("templates_gray must not be empty")
        self._cfg = cfg
        self._prep_cfg = MatchPrepConfig(use_clahe=cfg.use_clahe, blur_ksize=cfg.blur_ksize)
        self._templates: list[_PreparedTemplate] = []
        for template in templates_gray:
            self._templates.extend(self._prepare_template_variants(template))
        if not self._templates:
            raise ValueError("No valid scaled templates could be prepared.")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        detection = self.detect_best(frame)
        if detection is None:
            return []
        return [detection]

    def detect_best(self, frame: np.ndarray) -> Detection | None:
        return self.detect_best_with_stats(frame).detection

    def detect_best_with_stats(self, frame: np.ndarray) -> TemplateMatchResult:
        scene_gray, scene_edges = prepare_match_images(frame, self._prep_cfg)
        candidates: list[Detection] = []

        for template in self._templates:
            if template.height > scene_gray.shape[0] or template.width > scene_gray.shape[1]:
                continue

            response = self._combined_response(scene_gray, scene_edges, template)
            _, score, _, max_loc = cv.minMaxLoc(response)
            x, y = max_loc
            candidates.append(
                Detection(
                    label=self._cfg.label,
                    score=float(score),
                    bbox=BBox(int(x), int(y), int(x + template.width), int(y + template.height)),
                )
            )

            if self._cfg.min_score_margin > 0.0:
                suppressed = response.copy()
                self._suppress_response_peak(suppressed, x, y, template.width, template.height)
                _, second_score, _, second_loc = cv.minMaxLoc(suppressed)
                sx, sy = second_loc
                candidates.append(
                    Detection(
                        label=self._cfg.label,
                        score=float(second_score),
                        bbox=BBox(int(sx), int(sy), int(sx + template.width), int(sy + template.height)),
                    )
                )

        if not candidates:
            return TemplateMatchResult(None, -1.0, -1.0, 1.0, "no_valid_template")

        candidates.sort(key=lambda det: det.score, reverse=True)
        best = candidates[0]
        second_score = -1.0
        for candidate in candidates[1:]:
            if _bbox_iou(best.bbox, candidate.bbox) <= self._cfg.second_best_iou_threshold:
                second_score = candidate.score
                break

        margin = 1.0 if second_score < 0.0 else best.score - second_score
        if best.score < self._cfg.score_threshold:
            return TemplateMatchResult(None, best.score, second_score, margin, "low_score")
        if margin < self._cfg.min_score_margin:
            return TemplateMatchResult(None, best.score, second_score, margin, "ambiguous_score_margin")
        return TemplateMatchResult(best, best.score, second_score, margin)

    def detect_candidates(
        self,
        frame: np.ndarray,
        *,
        max_candidates: int = 8,
        score_threshold: float | None = None,
    ) -> list[Detection]:
        """Return several strong, spatially distinct template candidates."""
        scene_gray, scene_edges = prepare_match_images(frame, self._prep_cfg)
        threshold = self._cfg.score_threshold if score_threshold is None else score_threshold
        candidates: list[Detection] = []

        for template in self._templates:
            if template.height > scene_gray.shape[0] or template.width > scene_gray.shape[1]:
                continue

            response = self._combined_response(scene_gray, scene_edges, template).copy()
            peaks_per_template = max(1, min(3, max_candidates))
            for _ in range(peaks_per_template):
                _, score, _, max_loc = cv.minMaxLoc(response)
                if score < threshold:
                    break
                x, y = max_loc
                candidates.append(
                    Detection(
                        label=self._cfg.label,
                        score=float(score),
                        bbox=BBox(int(x), int(y), int(x + template.width), int(y + template.height)),
                    )
                )

                suppress_x1 = max(0, x - template.width // 2)
                suppress_y1 = max(0, y - template.height // 2)
                suppress_x2 = min(response.shape[1], x + template.width // 2)
                suppress_y2 = min(response.shape[0], y + template.height // 2)
                response[suppress_y1:suppress_y2, suppress_x1:suppress_x2] = -1.0

        candidates.sort(key=lambda det: det.score, reverse=True)
        return candidates[:max_candidates]

    def best_raw_score(self, frame: np.ndarray) -> float:
        """Return the best raw score without applying the configured threshold."""
        scene_gray, scene_edges = prepare_match_images(frame, self._prep_cfg)
        best = -1.0
        for template in self._templates:
            if template.height > scene_gray.shape[0] or template.width > scene_gray.shape[1]:
                continue
            response = self._combined_response(scene_gray, scene_edges, template)
            _, score, _, _ = cv.minMaxLoc(response)
            best = max(best, float(score))
        return best

    def _combined_response(self, scene_gray: np.ndarray, scene_edges: np.ndarray, template: _PreparedTemplate) -> np.ndarray:
        gray_response = cv.matchTemplate(scene_gray, template.gray, cv.TM_CCOEFF_NORMED)
        if self._cfg.edge_weight <= 0.0:
            return gray_response

        edge_response = cv.matchTemplate(scene_edges, template.edges, cv.TM_CCOEFF_NORMED)
        return self._cfg.gray_weight * gray_response + self._cfg.edge_weight * edge_response

    @staticmethod
    def _suppress_response_peak(response: np.ndarray, x: int, y: int, width: int, height: int) -> None:
        suppress_x1 = max(0, x - width // 2)
        suppress_y1 = max(0, y - height // 2)
        suppress_x2 = min(response.shape[1], x + width // 2 + 1)
        suppress_y2 = min(response.shape[0], y + height // 2 + 1)
        response[suppress_y1:suppress_y2, suppress_x1:suppress_x2] = -1.0

    def _prepare_template_variants(self, template: np.ndarray) -> list[_PreparedTemplate]:
        gray, edges = prepare_match_images(template, self._prep_cfg)
        variants: list[_PreparedTemplate] = []
        for scale in self._cfg.scales:
            if scale <= 0.0:
                continue
            width = int(round(gray.shape[1] * scale))
            height = int(round(gray.shape[0] * scale))
            if width < self._cfg.min_template_size or height < self._cfg.min_template_size:
                continue
            resized_gray = cv.resize(gray, (width, height), interpolation=cv.INTER_AREA)
            resized_edges = cv.resize(edges, (width, height), interpolation=cv.INTER_AREA)
            variants.append(_PreparedTemplate(resized_gray, resized_edges, width, height))
        return variants


def _bbox_iou(a: BBox, b: BBox) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union = a.area() + b.area() - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union
