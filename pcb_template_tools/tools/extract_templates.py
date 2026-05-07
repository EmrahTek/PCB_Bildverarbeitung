#!/usr/bin/env python3
"""
extract_templates.py

Select component ROIs once on a canonical warped board image, then extract the
same ROIs from multiple top-ranked warped images to build a stronger template bank.

This script is designed for the workflow:
1. run warp_and_rank_boards.py on iPhone images
2. choose top 3-5 warped boards (or load them automatically from the report)
3. select ROIs once on the first canonical board image
4. export templates for all selected source images

Because all warped inputs are already canonicalized to the same board coordinate
system, the same ROI coordinates can be reused across all selected board images.

Python docs:
- argparse: https://docs.python.org/3/library/argparse.html
- dataclasses: https://docs.python.org/3/library/dataclasses.html
- json: https://docs.python.org/3/library/json.html
- pathlib: https://docs.python.org/3/library/pathlib.html
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


DEFAULT_COMPONENTS = ["esp32", "usb_port", "jst_connector", "reset_button"]
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class TemplateRecord:
    component: str
    roi_xywh: list[int]
    source_image: str
    files: list[str]


# -----------------------------
# Image helpers
# -----------------------------
def ensure_gray(image: np.ndarray) -> np.ndarray:
    """Return a grayscale copy of a BGR or grayscale template image."""
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



def rotate_keep_size(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate an image around its center without changing canvas size."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )



def adjust_brightness(image: np.ndarray, alpha: float, beta: int = 0) -> np.ndarray:
    """Apply a simple brightness/contrast transform for template augmentation."""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)



def add_mild_noise(image: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Add light Gaussian noise to make the template bank less brittle."""
    noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)



def make_augmentations(template_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Create a small but useful augmentation set for template matching."""
    gray = ensure_gray(template_bgr)

    variants = {
        "base": gray,
        "bright": adjust_brightness(gray, alpha=1.12),
        "dark": adjust_brightness(gray, alpha=0.88),
        "blur": cv2.GaussianBlur(gray, (3, 3), 0),
        "rot_p5": rotate_keep_size(gray, 5.0),
        "rot_m5": rotate_keep_size(gray, -5.0),
        "noise": add_mild_noise(gray, sigma=4.0),
    }
    return variants


# -----------------------------
# ROI helpers
# -----------------------------
def parse_components(raw: str) -> list[str]:
    """Parse a comma-separated component list while preserving selection order."""
    components = [item.strip() for item in raw.split(",") if item.strip()]
    if not components:
        raise ValueError("Component list is empty. Provide at least one component name.")
    return components



def select_single_roi(window_name: str, image: np.ndarray) -> tuple[int, int, int, int]:
    """Open a selection window and return one ROI."""
    roi = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    x, y, w, h = map(int, roi)
    return x, y, w, h



def draw_labeled_boxes(image: np.ndarray, rois: dict[str, list[int]]) -> np.ndarray:
    """Draw selected ROI boxes and labels on a preview image."""
    preview = image.copy()
    for idx, (component, roi_xywh) in enumerate(rois.items(), start=1):
        x, y, w, h = roi_xywh
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            preview,
            f"{idx}:{component}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )
    return preview



def validate_roi(x: int, y: int, w: int, h: int, image_shape: tuple[int, ...], min_size: int) -> None:
    """Validate ROI size and image bounds before extracting templates."""
    if w == 0 or h == 0:
        raise ValueError("ROI width/height is zero.")
    if w < min_size or h < min_size:
        raise ValueError(
            f"ROI {w}x{h} is smaller than the minimum allowed size ({min_size}px)."
        )

    img_h, img_w = image_shape[:2]
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        raise ValueError(
            f"ROI [{x}, {y}, {w}, {h}] is outside image bounds {img_w}x{img_h}."
        )



def interactive_roi_selection(
    image: np.ndarray,
    components: Sequence[str],
    min_size: int,
) -> dict[str, list[int]]:
    """Collect component ROIs interactively from an OpenCV selection window."""
    print("[INFO] ROI extraction started.")
    print("[INFO] For each component, draw a rectangle and press ENTER or SPACE.")
    print("[INFO] Press 'c' in the ROI window to cancel a wrong selection and redraw.")
    print("[INFO] If you want to skip a component, press ESC and then close the selection window.")

    rois: dict[str, list[int]] = {}
    for component in components:
        window_name = f"Select ROI: {component}"
        x, y, w, h = select_single_roi(window_name, image)
        if w == 0 or h == 0:
            print(f"[WARN] Skipped component: {component}")
            continue
        validate_roi(x, y, w, h, image.shape, min_size)
        rois[component] = [x, y, w, h]
    return rois



def load_rois_from_file(roi_file: Path, image_shape: tuple[int, ...], min_size: int) -> dict[str, list[int]]:
    """Load and validate previously saved ROI coordinates."""
    with open(roi_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_rois = payload.get("rois") if isinstance(payload, dict) else None
    if not isinstance(raw_rois, dict) or not raw_rois:
        raise ValueError(f"ROI file has no valid 'rois' dictionary: {roi_file}")

    rois: dict[str, list[int]] = {}
    for component, coords in raw_rois.items():
        if not isinstance(coords, list) or len(coords) != 4:
            raise ValueError(f"Invalid ROI for component '{component}' in {roi_file}")
        x, y, w, h = map(int, coords)
        validate_roi(x, y, w, h, image_shape, min_size)
        rois[component] = [x, y, w, h]
    return rois


# -----------------------------
# Input-resolution helpers
# -----------------------------
def deduplicate_keep_order(paths: Sequence[Path]) -> list[Path]:
    """Return unique resolved paths without changing their first-seen order."""
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique



def resolve_images_from_report(report_path: Path, top_k: int) -> list[Path]:
    """Read the top ranked warped image paths from a quality report."""
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    all_results = report.get("all_results")
    if not isinstance(all_results, list) or not all_results:
        raise ValueError(f"Report does not contain a usable 'all_results' list: {report_path}")

    warped_paths: list[Path] = []
    for item in all_results[:top_k]:
        warped = item.get("warped")
        if isinstance(warped, str) and warped:
            warped_paths.append(Path(warped))

    if not warped_paths:
        raise ValueError(f"No warped image paths found in report: {report_path}")
    return deduplicate_keep_order(warped_paths)



def resolve_images(args: argparse.Namespace) -> list[Path]:
    """Merge image inputs from CLI flags and quality reports."""
    paths: list[Path] = []

    if args.image:
        paths.append(Path(args.image))

    if args.images:
        paths.extend(Path(item) for item in args.images)

    if args.report:
        paths.extend(resolve_images_from_report(Path(args.report), args.top_k))

    paths = deduplicate_keep_order(paths)
    if not paths:
        raise ValueError("Provide at least one input via --image, --images, or --report.")

    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max-images must be a positive integer")
        paths = paths[: args.max_images]

    return paths


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line options for template extraction."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract board-component templates from one or more canonical warped images. "
            "ROIs are selected once and reused across all images."
        )
    )
    parser.add_argument("--image", help="Single warped board image (legacy / simple mode)")
    parser.add_argument(
        "--images",
        nargs="+",
        help="One or more warped board images. Useful for building a multi-image template bank.",
    )
    parser.add_argument(
        "--report",
        help=(
            "Path to board_quality_report.json from warp_and_rank_boards.py. "
            "The script will load the top-k warped images from the ranking."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="How many top-ranked warped images to load from --report (default: 4)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Optional hard limit after merging --image / --images / --report inputs",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where template folders will be created")
    parser.add_argument(
        "--components",
        default=",".join(DEFAULT_COMPONENTS),
        help="Comma-separated component names in selection order",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=20,
        help="Minimum width/height for a valid ROI in pixels (default: 20)",
    )
    parser.add_argument(
        "--roi-file",
        help=(
            "Optional JSON file with previously saved ROIs. "
            "If provided, the script runs non-interactively and reuses those ROIs."
        ),
    )
    return parser.parse_args()



def main() -> None:
    """Run ROI selection and template export."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.min_size <= 0:
        raise ValueError("--min-size must be a positive integer")
    if args.top_k <= 0:
        raise ValueError("--top-k must be a positive integer")

    components = parse_components(args.components)
    image_paths = resolve_images(args)

    loaded_images: list[tuple[Path, np.ndarray]] = []
    expected_shape: tuple[int, ...] | None = None
    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image file type: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        if expected_shape is None:
            expected_shape = image.shape
        elif image.shape != expected_shape:
            raise ValueError(
                "All warped images must have identical shape for ROI reuse. "
                f"Expected {expected_shape}, got {image.shape} for {image_path}"
            )
        loaded_images.append((image_path, image))

    if not loaded_images:
        raise RuntimeError("No valid input images were loaded.")

    reference_path, reference_image = loaded_images[0]
    if args.roi_file:
        roi_file = Path(args.roi_file)
        if not roi_file.exists():
            raise FileNotFoundError(f"ROI file not found: {roi_file}")
        rois = load_rois_from_file(roi_file, reference_image.shape, args.min_size)
        print(f"[INFO] Loaded ROIs from: {roi_file}")
    else:
        rois = interactive_roi_selection(reference_image, components, args.min_size)

    cv2.destroyAllWindows()

    if not rois:
        raise RuntimeError("No valid ROIs are available. No templates were generated.")

    rois_path = output_dir / "component_rois.json"
    with open(rois_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "reference_image": str(reference_path),
                "reference_shape": list(reference_image.shape),
                "rois": rois,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    preview = draw_labeled_boxes(reference_image, rois)
    preview_path = output_dir / "selected_rois_preview.png"
    cv2.imwrite(str(preview_path), preview)

    records: list[TemplateRecord] = []
    total_files_written = 0
    for image_index, (image_path, image) in enumerate(loaded_images, start=1):
        stem_token = image_path.stem.replace(" ", "_")
        print(f"[INFO] Extracting from source {image_index}/{len(loaded_images)}: {image_path.name}")

        for component, roi_xywh in rois.items():
            x, y, w, h = roi_xywh
            crop = image[y:y + h, x:x + w]
            if crop.size == 0:
                raise RuntimeError(
                    f"ROI for component '{component}' produced an empty crop on image {image_path}"
                )

            comp_dir = output_dir / component
            comp_dir.mkdir(parents=True, exist_ok=True)

            variants = make_augmentations(crop)
            saved_files: list[str] = []
            prefix = f"{component}_img{image_index:02d}_{stem_token}"
            for variant_name, variant_img in variants.items():
                out_path = comp_dir / f"{prefix}_{variant_name}.png"
                cv2.imwrite(str(out_path), variant_img)
                saved_files.append(str(out_path))
                total_files_written += 1

            records.append(
                TemplateRecord(
                    component=component,
                    roi_xywh=[x, y, w, h],
                    source_image=str(image_path),
                    files=saved_files,
                )
            )
            print(f"[OK] {component}: saved {len(saved_files)} file(s) from {image_path.name}")

    metadata = {
        "mode": "multi_image_canonical_roi_reuse",
        "num_source_images": len(loaded_images),
        "source_images": [str(path) for path, _ in loaded_images],
        "reference_image": str(reference_path),
        "image_shape": list(reference_image.shape),
        "components": list(rois.keys()),
        "rois": rois,
        "records": [asdict(record) for record in records],
        "preview": str(preview_path),
        "roi_file": str(rois_path),
    }
    metadata_path = output_dir / "templates_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[INFO] ROI file written to: {rois_path}")
    print(f"[INFO] Template metadata written to: {metadata_path}")
    print(f"[INFO] ROI preview written to: {preview_path}")
    print(f"[INFO] Templates saved under: {output_dir}")
    print(
        f"[INFO] Finished. Generated {total_files_written} template image(s) "
        f"from {len(loaded_images)} source board image(s)."
    )


if __name__ == "__main__":
    main()
