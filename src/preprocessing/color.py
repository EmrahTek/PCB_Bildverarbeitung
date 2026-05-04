"""Color conversion helpers for OpenCV frames.

This module normalizes image inputs into grayscale uint8 arrays used by the
geometry and template-matching stages.

Python docs:
- exceptions: https://docs.python.org/3/tutorial/errors.html
"""

from __future__ import annotations

import cv2 as cv
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert either grayscale or BGR input to grayscale."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def normalize_gray(gray: np.ndarray) -> np.ndarray:
    """Normalize grayscale arrays into uint8 range [0, 255]."""
    if gray.dtype == np.uint8:
        return gray
    return cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
