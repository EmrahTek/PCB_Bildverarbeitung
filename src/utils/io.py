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

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def project_root() -> Path:
    """Return the repository root when called from src/utils/io.py."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML configuration files") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML top-level object must be a dictionary: {path}")
    return data


def list_image_files(directory: Path, *, recursive: bool = False) -> list[Path]:
    """List image files in a directory, sorted by filename."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")
    pattern = "**/*" if recursive else "*"
    files = [p for p in directory.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    return sorted(files, key=lambda p: p.name.lower())


def load_bgr(path: Path) -> np.ndarray:
    """Load an image in OpenCV BGR format."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv.imread(str(path), cv.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")
    return img


def load_gray(path: Path) -> np.ndarray:
    """Load an image as grayscale uint8."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")
    return img


def load_templates(template_dir: Path, *, recursive: bool = False, limit: int | None = None) -> list[np.ndarray]:
    """Load template images from disk as grayscale arrays."""
    paths = list_image_files(template_dir, recursive=recursive)
    if limit is not None:
        paths = paths[:limit]
    return [load_gray(path) for path in paths]


def save_debug_image(path: Path, image: np.ndarray) -> None:
    """Save a debug image and create the parent directory if needed."""
    ensure_dir(path.parent)
    ok = cv.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Could not save image to: {path}")
