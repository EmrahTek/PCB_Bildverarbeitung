from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.utils.io import load_bgr


_LABEL_ALIASES = {
    "esp32": "ESP32",
    "usb_port": "USB_PORT",
    "jst_connector": "JST_CONNECTOR",
    "reset_button": "RESET_BUTTON",
}


@dataclass(frozen=True)
class PreparedTemplateBank:
    """Resolved template-bank metadata exported by pcb_template_tools."""

    metadata_path: Path
    root_dir: Path
    source_image: Path
    board_dir: Path
    canonical_size: tuple[int, int]
    component_dirs: dict[str, Path]
    component_rois: dict[str, tuple[float, float, float, float]]


def _resolve_relative_path(root_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        if path.parts and path.parts[0] == root_dir.name:
            return root_dir / Path(*path.parts[1:])
        return root_dir / path

    # The preparation tools may have been run from a different clone path. Keep
    # the bank portable by remapping absolute paths that contain pcb_template_tools
    # back into the current metadata root when possible.
    parts = path.parts
    if "pcb_template_tools" in parts:
        suffix = Path(*parts[parts.index("pcb_template_tools") + 1 :])
        remapped = root_dir / suffix
        if remapped.exists():
            return remapped
    return path


def _metadata_root(metadata_path: Path) -> Path:
    for parent in metadata_path.parents:
        if parent.name == "pcb_template_tools":
            return parent
    return metadata_path.parents[2]


def rotate_bbox_xywh(
    x: int,
    y: int,
    width: int,
    height: int,
    src_width: int,
    src_height: int,
    turns_90: int,
) -> tuple[int, int, int, int]:
    """Rotate an XYWH box by multiples of 90 degrees using continuous image bounds."""
    turns = turns_90 % 4
    if turns == 0:
        return x, y, width, height
    if turns == 1:
        return src_height - (y + height), x, height, width
    if turns == 2:
        return src_width - (x + width), src_height - (y + height), width, height
    return y, src_width - (x + width), height, width


def rotated_size(src_width: int, src_height: int, turns_90: int) -> tuple[int, int]:
    """Return image size after rotating by multiples of 90 degrees."""
    if turns_90 % 2 == 0:
        return src_width, src_height
    return src_height, src_width


def load_prepared_template_bank(metadata_path: Path, rotation_turns: int = 0) -> PreparedTemplateBank:
    """Load pcb_template_tools metadata and convert it into runtime-friendly paths and ROIs."""
    metadata_path = metadata_path.resolve()
    root_dir = _metadata_root(metadata_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    source_key = "source_image" if "source_image" in payload else "reference_image"
    source_image = _resolve_relative_path(root_dir, str(payload[source_key])).resolve()
    source = load_bgr(source_image)
    src_height, src_width = source.shape[:2]
    canonical_width, canonical_height = rotated_size(src_width, src_height, rotation_turns)

    component_dirs: dict[str, Path] = {}
    component_rois: dict[str, tuple[float, float, float, float]] = {}
    component_payloads = payload.get("records")
    if not isinstance(component_payloads, list):
        component_payloads = payload.get("components", [])

    for component_payload in component_payloads:
        if not isinstance(component_payload, dict):
            continue
        raw_label = str(component_payload.get("component", "")).lower()
        label = _LABEL_ALIASES.get(raw_label)
        files = component_payload.get("files", [])
        roi_xywh = component_payload.get("roi_xywh", [])
        if label is None or not files or len(roi_xywh) != 4:
            continue

        template_path = _resolve_relative_path(root_dir, str(files[0])).resolve()
        x, y, width, height = (int(value) for value in roi_xywh)
        rx, ry, rw, rh = rotate_bbox_xywh(
            x,
            y,
            width,
            height,
            src_width,
            src_height,
            rotation_turns,
        )
        component_dirs[label] = template_path.parent
        component_rois[label] = (
            rx / canonical_width,
            ry / canonical_height,
            (rx + rw) / canonical_width,
            (ry + rh) / canonical_height,
        )

    return PreparedTemplateBank(
        metadata_path=metadata_path,
        root_dir=root_dir,
        source_image=source_image,
        board_dir=source_image.parent,
        canonical_size=(canonical_width, canonical_height),
        component_dirs=component_dirs,
        component_rois=component_rois,
    )
