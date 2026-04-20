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


@dataclass(frozen=True)
class _PreparedTemplate:
    gray: np.ndarray
    edges: np.ndarray
    width: int
    height: int


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
        scene_gray, scene_edges = prepare_match_images(frame, self._prep_cfg)
        best: Detection | None = None

        for template in self._templates:
            if template.height > scene_gray.shape[0] or template.width > scene_gray.shape[1]:
                continue

            response = self._combined_response(scene_gray, scene_edges, template)
            _, score, _, max_loc = cv.minMaxLoc(response)
            if best is None or score > best.score:
                x, y = max_loc
                best = Detection(
                    label=self._cfg.label,
                    score=float(score),
                    bbox=BBox(int(x), int(y), int(x + template.width), int(y + template.height)),
                )

        if best is None or best.score < self._cfg.score_threshold:
            return None
        return best

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
