from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.preprocessing.color import normalize_gray, to_gray


@dataclass(frozen=True)
class MatchPrepConfig:
    """Configuration for template matching preprocessing."""
    use_clahe: bool = True
    blur_ksize: int = 3


def gaussian_blur(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply Gaussian blur with a guaranteed odd kernel size."""
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    if k == 1:
        return image
    return cv.GaussianBlur(image, (k, k), 0)


def clahe_gray(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE to reduce sensitivity to uneven lighting."""
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def canny_edges(gray: np.ndarray, t1: int = 45, t2: int = 135) -> np.ndarray:
    """Extract Canny edges from a grayscale image."""
    return cv.Canny(gray, t1, t2)


def prepare_match_images(image: np.ndarray, cfg: MatchPrepConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare grayscale and edge representations for template matching.

    The detector combines both signals to stay reasonably robust under lighting
    changes while still keeping enough texture information for components.
    """
    gray = normalize_gray(to_gray(image))
    if cfg.use_clahe:
        gray = clahe_gray(gray)
    gray = gaussian_blur(gray, cfg.blur_ksize)
    edges = canny_edges(gray)
    return gray, edges
