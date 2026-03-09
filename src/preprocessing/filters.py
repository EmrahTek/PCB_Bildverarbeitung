# Wiederverwendbare Filter-Bausteine: Blur, CLAHE, Canny, Morphology.

"""
filters.py

This module provides reusable image filtering primitives such as:
- Gaussian blur
- CLAHE contrast enhancement
- Canny edge detection
- Optional morphological operators

Inputs:
- Grayscale or BGR images as NumPy arrays

Outputs:
- Filtered images (grayscale, edges, binary masks) as NumPy arrays

Zu implementierende Funktionen

    gaussian_blur(img, ksize=(5,5), sigma=0) -> img

    apply_clahe(gray, clip_limit=2.0, tile_grid_size=(8,8)) -> gray

    canny_edges(gray, t1, t2) -> edges

    morph_open(binary, ksize=(3,3), iterations=1) -> binary (optional)

GaussianBlur:
https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

CLAHE:
https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html

Canny:
https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html

Morphology:
https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

"""

# src/preprocessing/filters.py
from __future__ import annotations

import cv2 as cv
import numpy as np


def gaussian_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply Gaussian blur with odd kernel size."""
    k = max(3, ksize)
    if k % 2 == 0:
        k += 1
    return cv.GaussianBlur(img, (k, k), 0)


def clahe_gray(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE on grayscale image to improve local contrast under varying illumination.
    """
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def canny_edges(gray: np.ndarray, t1: int = 50, t2: int = 150) -> np.ndarray:
    """Compute Canny edges."""
    return cv.Canny(gray, t1, t2)