# save frames, load templates, paths

# Alle Datei-/Pfad-Operationen: YAML-Konfig laden, Templates laden, Debug-Frames speichern, Assets auflisten.

"""
io.py

This module contains filesystem and I/O helpers for the project:
- loading YAML configuration files
- resolving project-relative asset paths
- ensuring directories exist
- saving debug images
- loading template images from disk

Inputs:
- Paths (pathlib.Path) to config files, assets, or output folders.
- NumPy arrays representing images (OpenCV format).

Outputs:
- Python dictionaries (from YAML config)
- Template image collections (e.g., dict[label] -> list[np.ndarray])
- Debug images saved to disk

Zu implementierende Funktionen

project_root() -> Path

load_yaml(path: Path) -> dict

resolve_asset_path(*parts) -> Path

ensure_dir(path: Path) -> None

save_debug_image(path: Path, image: np.ndarray) -> None

list_images(folder: Path, exts=(".png",".jpg",".jpeg")) -> list[Path]

load_templates(folder: Path) -> dict[str, list[np.ndarray]] (label -> templates)

pathlib:
https://docs.python.org/3/library/pathlib.html

PyYAML (safe_load):
https://pyyaml.org/wiki/PyYAMLDocumentation

NumPy array basics (for images):
https://numpy.org/doc/stable/user/quickstart.html




"""


# src/utils/io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np


# Supported image extensions for templates / test images
SUPPORTED_IMAGE_EXTS: set[str] = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}


@dataclass(frozen=True)
class LoadedImage:
    """Container holding image data and its source path."""
    path: Path
    image: np.ndarray


def list_image_files(directory: Path, *, recursive: bool = False) -> list[Path]:
    """
    List image files in a directory, optionally recursively.

    Args:
        directory: Directory containing images.
        recursive: If True, search subdirectories.

    Returns:
        Sorted list of image paths.

    Raises:
        FileNotFoundError: If directory does not exist.
        NotADirectoryError: If path exists but is not a directory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    files = [
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def load_bgr(path: Path) -> np.ndarray:
    """
    Load an image from disk as BGR (OpenCV default).

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If OpenCV fails to decode the image.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv.imread(str(path), cv.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale uint8.

    Accepts:
        - gray images (H, W)
        - BGR images  (H, W, 3)

    Raises:
        ValueError: If shape is unsupported.
    """
    if img.ndim == 2:
        out = img
    elif img.ndim == 3 and img.shape[2] == 3:
        out = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape for gray conversion: {img.shape}")

    if out.dtype != np.uint8:
        out = cv.normalize(out, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    return out


def load_images(
    directory: Path,
    *,
    as_gray: bool = True,
    recursive: bool = False,
    limit: int | None = None,
) -> list[LoadedImage]:
    """
    Load all images from a directory.

    Args:
        directory: Folder containing images.
        as_gray: If True, convert each image to grayscale.
        recursive: If True, search subfolders as well.
        limit: Optional maximum number of images to load.

    Returns:
        List of LoadedImage(path, image).

    Raises:
        ValueError: If no images found.
    """
    paths = list_image_files(directory, recursive=recursive)
    if not paths:
        raise ValueError(f"No image files found in: {directory}")

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be a positive integer.")
        paths = paths[:limit]

    loaded: list[LoadedImage] = []
    for p in paths:
        bgr = load_bgr(p)
        img = to_gray(bgr) if as_gray else bgr
        loaded.append(LoadedImage(path=p, image=img))

    return loaded


def load_templates(
    template_dir: Path,
    *,
    recursive: bool = False,
    limit: int | None = None,
) -> list[np.ndarray]:
    """
    Convenience helper for template matching.

    Loads templates as GRAYSCALE and returns only the image arrays.

    Args:
        template_dir: Directory containing template images.
        recursive: If True, includes subfolders.
        limit: Optional maximum number of templates.

    Returns:
        List of grayscale template images.
    """
    return [
        item.image
        for item in load_images(template_dir, as_gray=True, recursive=recursive, limit=limit)
    ]
