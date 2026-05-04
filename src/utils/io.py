"""Filesystem and image-loading utilities.

This module resolves project paths, loads YAML/image/template files, samples
lists deterministically, and writes debug images.

Python docs:
- pathlib: https://docs.python.org/3/library/pathlib.html
- typing: https://docs.python.org/3/library/typing.html
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def project_root() -> Path:
    """Return the repository root when called from inside the src tree."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist yet."""
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML configuration files.") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dictionary: {path}")
    return data


def list_image_files(directory: Path, *, recursive: bool = False) -> list[Path]:
    """Return all supported image files in sorted order."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    paths = [path for path in directory.glob(pattern) if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    return sorted(paths, key=lambda path: path.name.lower())


def first_existing_directory(candidates: list[str | Path]) -> Path | None:
    """Return the first directory that exists from a list of candidate paths."""
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.is_dir():
            return path
    return None


def load_bgr(path: Path) -> np.ndarray:
    """Load an image in OpenCV BGR format."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def load_gray(path: Path) -> np.ndarray:
    """Load an image as grayscale uint8."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def load_templates(template_dir: Path, *, recursive: bool = False, limit: int | None = None) -> list[np.ndarray]:
    """Load a directory of template images as grayscale arrays."""
    paths = list_image_files(template_dir, recursive=recursive)
    if limit is not None:
        paths = paths[:limit]
    return [load_gray(path) for path in paths]


def sample_evenly(items: list[Any], count: int) -> list[Any]:
    """Sample a list evenly without requiring random state."""
    if count <= 0:
        raise ValueError("count must be positive")
    if len(items) <= count:
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=count, dtype=int)
    return [items[int(index)] for index in indices]


def rotate_image(image: np.ndarray, turns_90: int) -> np.ndarray:
    """Rotate an image by multiples of 90 degrees."""
    turns = turns_90 % 4
    if turns == 0:
        return image.copy()
    if turns == 1:
        return cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    if turns == 2:
        return cv.rotate(image, cv.ROTATE_180)
    return cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)


def save_debug_image(path: Path, image: np.ndarray) -> None:
    """Save a debug image and create its parent folder if needed."""
    ensure_dir(path.parent)
    ok = cv.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Could not save image: {path}")
