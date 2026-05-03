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
    mode: str = "default"


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


def unsharp_mask(gray: np.ndarray, amount: float = 0.45, ksize: int = 5) -> np.ndarray:
    """Apply a small unsharp mask to recover local component edges."""
    blurred = gaussian_blur(gray, ksize)
    return cv.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)


def local_brightness_normalize(gray: np.ndarray, ksize: int = 31) -> np.ndarray:
    """Suppress slow illumination changes while keeping local contrast."""
    k = max(3, int(ksize))
    if k % 2 == 0:
        k += 1
    background = cv.GaussianBlur(gray, (k, k), 0)
    normalized = cv.addWeighted(gray, 1.0, background, -0.45, 64)
    return normalize_gray(normalized)


def prepare_match_images(image: np.ndarray, cfg: MatchPrepConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare grayscale and edge representations for template matching.

    The detector combines both signals to stay reasonably robust under lighting
    changes while still keeping enough texture information for components.
    """
    gray = normalize_gray(to_gray(image))
    if cfg.use_clahe:
        gray = clahe_gray(gray)

    mode = cfg.mode.lower()
    if mode in {"module", "metal", "connector", "button"}:
        gray = local_brightness_normalize(gray, 31)
        gray = clahe_gray(gray, clip_limit=2.4)
    if mode in {"module", "metal", "button"}:
        gray = unsharp_mask(gray, amount=0.40, ksize=5)
    if mode == "connector":
        top_hat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, np.ones((9, 9), np.uint8))
        gray = cv.addWeighted(gray, 0.82, top_hat, 0.75, 0)
        gray = normalize_gray(gray)
    if mode == "button":
        kernel = np.ones((5, 5), np.uint8)
        top_hat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
        black_hat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
        gray = cv.addWeighted(gray, 0.78, top_hat, 0.46, 0)
        gray = cv.addWeighted(gray, 1.00, black_hat, 0.34, 0)
        gray = normalize_gray(gray)
    if mode == "metal":
        grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
        grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
        grad = cv.convertScaleAbs(cv.magnitude(grad_x, grad_y))
        gray = cv.addWeighted(gray, 0.76, grad, 0.38, 0)
        gray = normalize_gray(gray)

    gray = gaussian_blur(gray, cfg.blur_ksize)
    edges = canny_edges(gray)
    return gray, edges
