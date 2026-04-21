#!/usr/bin/env python3
"""
warp_and_rank_boards.py

Batch-process raw PCB photos, detect the board on a white A4 sheet,
warp each valid board into a canonical top-down view, normalize orientation,
compute quality metrics, and rank the results.

Recommended capture conditions for this script:
- PCB placed on white A4 paper
- iPhone shot from above
- daylight or soft diffuse light
- board fully visible in the frame

Canonical output convention for this toolkit:
- warped image size is fixed (default: 900 x 460)
- board long edge is horizontal
- ESP32/BLE module should end up on the LEFT
- USB + JST should end up on the RIGHT

The orientation normalization heuristic is intentionally board-specific and tuned
for the FireBeetle / ESP32-style board used in this project.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class QualityMetrics:
    sharpness: float
    contrast: float
    brightness_mean: float
    brightness_std: float
    exposure_score: float
    saturation_penalty: float
    board_fill_ratio: float
    total_score: float


@dataclass
class DetectionResult:
    source: str
    warped: str
    preview: str
    mask: str
    board_found: bool
    detection_method: str
    board_area_ratio: float
    aspect_ratio: float
    canonical_rotation_deg: int
    canonical_right_minus_left_brightness: float
    metrics: QualityMetrics


@dataclass
class FailedResult:
    source: str
    reason: str


# -----------------------------
# JSON helpers
# -----------------------------
def json_default(obj):
    """Convert NumPy scalar types to native Python types for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# -----------------------------
# Numeric helpers
# -----------------------------
def normalized_aspect(value: float) -> float:
    """
    Make aspect ratio rotation-invariant.
    Example:
        2.0   -> 0.5
        0.5   -> 0.5
        1.8   -> 0.555...
    """
    value = max(float(value), 1e-6)
    return min(value, 1.0 / value)


# -----------------------------
# Geometry helpers
# -----------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    """Return the 4 points in TL, TR, BR, BL order."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected shape (4, 2), got {pts.shape}")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect


def warp_from_quad(image: np.ndarray, quad: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Perspective-warp the board into a canonical output size."""
    rect = order_points(quad)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)


# -----------------------------
# Canonical orientation helpers
# -----------------------------
def compute_left_right_brightness_delta(
    image_bgr: np.ndarray,
    side_fraction: float = 0.18,
    vertical_margin_fraction: float = 0.08,
) -> float:
    """
    Compute right-minus-left brightness on side bands.

    For this board family, the USB + JST side is typically brighter than the
    ESP32 antenna/module side. That makes this a practical board-specific
    heuristic for deciding whether a 180-degree rotation is needed.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    band_w = max(10, int(round(w * side_fraction)))
    y0 = max(0, int(round(h * vertical_margin_fraction)))
    y1 = min(h, int(round(h * (1.0 - vertical_margin_fraction))))
    if y1 <= y0:
        y0, y1 = 0, h

    roi = gray[y0:y1, :]
    left_mean = float(roi[:, :band_w].mean())
    right_mean = float(roi[:, w - band_w:].mean())
    return right_mean - left_mean


def canonicalize_warped_orientation(image_bgr: np.ndarray) -> tuple[np.ndarray, int, float]:
    """
    Enforce a fixed board orientation.

    Rule:
    - keep image as-is if the right side is brighter than the left side
    - otherwise rotate 180 degrees

    This is intentionally simple and board-specific. The project goal is a
    stable canonical reference for one known PCB, not a universal PCB parser.
    """
    delta = compute_left_right_brightness_delta(image_bgr)
    if delta >= 0.0:
        return image_bgr, 0, float(round(delta, 3))

    rotated = cv2.rotate(image_bgr, cv2.ROTATE_180)
    rotated_delta = compute_left_right_brightness_delta(rotated)
    return rotated, 180, float(round(rotated_delta, 3))


# -----------------------------
# Detection helpers
# -----------------------------
def resize_for_detection(image: np.ndarray, max_side: int = 1800) -> tuple[np.ndarray, float]:
    """Resize large images to speed up detection while keeping aspect ratio."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image.copy(), 1.0

    scale = max_side / float(longest)
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def is_valid_quad(
    quad: np.ndarray,
    image_shape: tuple[int, int, int],
    expected_aspect: float,
) -> tuple[bool, float, float]:
    """
    Validate quadrilateral using area and rotation-invariant aspect-ratio heuristics.
    """
    h, w = image_shape[:2]
    image_area = float(h * w)
    area = float(cv2.contourArea(quad.astype(np.float32)))
    area_ratio = area / image_area if image_area > 0 else 0.0

    if area_ratio < 0.03:
        return False, area_ratio, 0.0

    ordered = order_points(quad.astype(np.float32))
    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bottom = np.linalg.norm(ordered[2] - ordered[3])
    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])

    width = max((width_top + width_bottom) / 2.0, 1e-6)
    height = max((height_left + height_right) / 2.0, 1e-6)
    aspect = width / height

    aspect_n = normalized_aspect(aspect)
    expected_n = normalized_aspect(expected_aspect)

    # More stable tolerance for rotated / portrait / landscape captures.
    aspect_ok = 0.70 * expected_n <= aspect_n <= 1.30 * expected_n
    area_ok = area_ratio >= 0.04
    return (aspect_ok and area_ok), area_ratio, aspect


def detect_board_quad(
    image: np.ndarray,
    expected_aspect: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str, float, float]:
    """
    Detect the PCB contour in a raw photo.

    Strategy:
    1. Detect non-white objects on white A4 paper.
    2. Clean mask morphologically.
    3. Keep only the largest connected component.
    4. Find the best 4-corner candidate.
    5. Fallback to rotated rectangle if a perfect quad is not found.
    """
    resized, scale = resize_for_detection(image)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # White paper usually has high V and low S.
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 155], dtype=np.uint8),
        np.array([180, 95, 255], dtype=np.uint8),
    )
    non_white = cv2.bitwise_not(white_mask)

    # Combine with a mild gradient signal so the board border stands out better.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 130)
    combined = cv2.bitwise_or(non_white, edges)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    # Keep only the largest connected component.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask, "none", 0.0, 0.0

    resized_area = float(resized.shape[0] * resized.shape[1])
    best_quad = None
    best_method = "none"
    best_score = -math.inf
    best_area_ratio = 0.0
    best_aspect = 0.0

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(cnt))
        if area < 0.02 * resized_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            valid, area_ratio, aspect = is_valid_quad(quad, resized.shape, expected_aspect)
            if valid:
                aspect_penalty = abs(
                    math.log(
                        normalized_aspect(max(aspect, 1e-6)) /
                        normalized_aspect(expected_aspect)
                    )
                )
                score = area_ratio * 100.0 - 8.0 * aspect_penalty
                if score > best_score:
                    best_score = score
                    best_quad = quad
                    best_method = "approx_quad"
                    best_area_ratio = area_ratio
                    best_aspect = aspect

    # Fallback: use minimum-area rectangle if no clean quad was found.
    if best_quad is None:
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            area = float(cv2.contourArea(cnt))
            if area < 0.03 * resized_area:
                continue

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.float32)
            valid, area_ratio, aspect = is_valid_quad(box, resized.shape, expected_aspect)
            if valid:
                aspect_penalty = abs(
                    math.log(
                        normalized_aspect(max(aspect, 1e-6)) /
                        normalized_aspect(expected_aspect)
                    )
                )
                score = area_ratio * 100.0 - 10.0 * aspect_penalty
                if score > best_score:
                    best_score = score
                    best_quad = box
                    best_method = "min_area_rect"
                    best_area_ratio = area_ratio
                    best_aspect = aspect

    if best_quad is None:
        return None, mask, "none", 0.0, 0.0

    if scale != 1.0:
        best_quad = best_quad / scale

    return best_quad.astype(np.float32), mask, best_method, best_area_ratio, best_aspect


# -----------------------------
# Quality helpers
# -----------------------------
def compute_quality_metrics(image_bgr: np.ndarray) -> QualityMetrics:
    """Compute image-quality metrics on the warped board image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Sharpness: higher is usually better for template creation.
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Contrast measures spread of intensity values.
    contrast = float(gray.std())

    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())

    # Exposure score: prefer brightness near a mid-high range rather than too dark or too clipped.
    target_brightness = 145.0
    exposure_score = 1.0 - abs(brightness_mean - target_brightness) / 145.0
    exposure_score = float(np.clip(exposure_score, 0.0, 1.0))

    # Penalize very large saturated areas.
    dark_ratio = float(np.mean(gray < 15))
    bright_ratio = float(np.mean(gray > 245))
    saturation_penalty = float(np.clip((dark_ratio + bright_ratio) * 2.0, 0.0, 1.0))

    # Estimate useful fill ratio from non-background pixels in the warped board.
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 155], dtype=np.uint8),
        np.array([180, 95, 255], dtype=np.uint8),
    )
    board_fill_ratio = 1.0 - float(np.mean(white_mask > 0))
    board_fill_ratio = float(np.clip(board_fill_ratio, 0.0, 1.0))

    # Weighted total score. The weights are chosen for template generation, not for aesthetics.
    total_score = (
        0.52 * min(sharpness / 300.0, 1.0) +
        0.18 * min(contrast / 80.0, 1.0) +
        0.14 * exposure_score +
        0.10 * board_fill_ratio +
        0.06 * (1.0 - saturation_penalty)
    ) * 100.0

    return QualityMetrics(
        sharpness=round(sharpness, 3),
        contrast=round(contrast, 3),
        brightness_mean=round(brightness_mean, 3),
        brightness_std=round(brightness_std, 3),
        exposure_score=round(exposure_score, 4),
        saturation_penalty=round(saturation_penalty, 4),
        board_fill_ratio=round(board_fill_ratio, 4),
        total_score=round(total_score, 3),
    )


# -----------------------------
# Processing pipeline
# -----------------------------
def save_debug_preview(
    image: np.ndarray,
    quad: Optional[np.ndarray],
    preview_path: Path,
    label: str,
) -> None:
    """Save original image with overlaid board polygon and a status label."""
    preview = image.copy()

    if quad is not None:
        cv2.polylines(preview, [quad.astype(np.int32)], True, (0, 255, 0), 4)
        for idx, pt in enumerate(order_points(quad), start=1):
            p = tuple(pt.astype(int))
            cv2.circle(preview, p, 8, (0, 0, 255), -1)
            cv2.putText(
                preview,
                str(idx),
                (p[0] + 8, p[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

    cv2.putText(
        preview,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    cv2.imwrite(str(preview_path), preview)


def save_mask(mask: Optional[np.ndarray], mask_path: Path, fallback_shape: tuple[int, int]) -> None:
    """Save detection mask for debugging."""
    if mask is None:
        blank = np.zeros(fallback_shape, dtype=np.uint8)
        cv2.imwrite(str(mask_path), blank)
    else:
        cv2.imwrite(str(mask_path), mask)


def iter_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def process_single_image(
    image_path: Path,
    output_warped_dir: Path,
    output_previews_dir: Path,
    output_masks_dir: Path,
    out_w: int,
    out_h: int,
) -> DetectionResult | FailedResult:
    image = cv2.imread(str(image_path))
    if image is None:
        return FailedResult(source=str(image_path), reason="could_not_read_image")

    expected_aspect = out_w / float(out_h)
    quad, mask, method, area_ratio, aspect_ratio = detect_board_quad(image, expected_aspect)

    preview_path = output_previews_dir / f"{image_path.stem}_preview.png"
    mask_path = output_masks_dir / f"{image_path.stem}_mask.png"

    if quad is None:
        save_debug_preview(image, None, preview_path, "DETECTION FAILED")
        save_mask(mask, mask_path, image.shape[:2])
        return FailedResult(source=str(image_path), reason="board_not_found")

    warped_raw = warp_from_quad(image, quad, out_w, out_h)
    warped, canonical_rotation_deg, brightness_delta = canonicalize_warped_orientation(warped_raw)
    metrics = compute_quality_metrics(warped)

    warped_path = output_warped_dir / f"{image_path.stem}_warped.png"
    cv2.imwrite(str(warped_path), warped)
    save_debug_preview(
        image,
        quad,
        preview_path,
        f"FOUND: {method} | canonical_rot={canonical_rotation_deg}",
    )
    save_mask(mask, mask_path, image.shape[:2])

    return DetectionResult(
        source=str(image_path),
        warped=str(warped_path),
        preview=str(preview_path),
        mask=str(mask_path),
        board_found=True,
        detection_method=method,
        board_area_ratio=float(round(area_ratio, 4)),
        aspect_ratio=float(round(aspect_ratio, 4)),
        canonical_rotation_deg=canonical_rotation_deg,
        canonical_right_minus_left_brightness=brightness_delta,
        metrics=metrics,
    )


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warp raw board photos, normalize to canonical orientation, and rank the results by quality."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing raw board photos")
    parser.add_argument("--output-dir", required=True, help="Directory for all generated outputs")
    parser.add_argument("--width", type=int, default=900, help="Warped board width (default: 900)")
    parser.add_argument("--height", type=int, default=460, help="Warped board height (default: 460)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive integers")

    image_paths = list(iter_images(input_dir))
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in: {input_dir}")

    warped_dir = output_dir / "warped"
    previews_dir = output_dir / "previews"
    masks_dir = output_dir / "masks"
    warped_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    successes: list[DetectionResult] = []
    failures: list[FailedResult] = []

    print(f"[INFO] Found {len(image_paths)} input image(s).")
    for image_path in image_paths:
        print(f"[INFO] Processing: {image_path.name}")
        result = process_single_image(
            image_path=image_path,
            output_warped_dir=warped_dir,
            output_previews_dir=previews_dir,
            output_masks_dir=masks_dir,
            out_w=args.width,
            out_h=args.height,
        )

        if isinstance(result, FailedResult):
            failures.append(result)
            print(f"[WARN] {image_path.name}: {result.reason}")
        else:
            successes.append(result)
            print(
                f"[OK] {image_path.name}: method={result.detection_method}, "
                f"canonical_rot={result.canonical_rotation_deg}, "
                f"score={result.metrics.total_score:.2f}, sharpness={result.metrics.sharpness:.2f}"
            )

    if not successes:
        report = {
            "num_input_images": len(image_paths),
            "num_successes": 0,
            "num_failures": len(failures),
            "failures": [asdict(item) for item in failures],
            "best_image": None,
            "all_results": [],
        }
        report_path = output_dir / "board_quality_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=json_default)

        raise RuntimeError(
            "No valid warped boards were produced. Check previews/masks, and verify that the board is placed on white A4 and fully visible."
        )

    successes_sorted = sorted(successes, key=lambda item: item.metrics.total_score, reverse=True)
    best = successes_sorted[0]

    report = {
        "num_input_images": len(image_paths),
        "num_successes": len(successes),
        "num_failures": len(failures),
        "best_image": {
            **asdict(best),
            "metrics": asdict(best.metrics),
        },
        "all_results": [
            {
                **asdict(item),
                "metrics": asdict(item.metrics),
            }
            for item in successes_sorted
        ],
        "failures": [asdict(item) for item in failures],
    }

    report_path = output_dir / "board_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=json_default)

    print("\n=== BEST WARPED IMAGE ===")
    print(json.dumps(report["best_image"], indent=2, ensure_ascii=False, default=json_default))
    print(f"\n[INFO] Report written to: {report_path}")
    print(f"[INFO] Warped images: {warped_dir}")
    print(f"[INFO] Preview overlays: {previews_dir}")
    print(f"[INFO] Debug masks: {masks_dir}")


if __name__ == "__main__":
    main()
