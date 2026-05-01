from __future__ import annotations

import json
from pathlib import Path

import cv2 as cv
import numpy as np

from src.utils.prepared_templates import load_prepared_template_bank, rotate_bbox_xywh


def test_rotate_bbox_xywh_clockwise() -> None:
    rotated = rotate_bbox_xywh(118, 2, 120, 116, src_width=460, src_height=900, turns_90=1)
    assert rotated == (782, 118, 116, 120)


def test_load_prepared_template_bank_rotates_roi_and_resolves_paths(tmp_path: Path) -> None:
    root = tmp_path / "pcb_template_tools"
    source_dir = root / "data" / "preparation_output_fixed" / "warped"
    template_dir = root / "assets" / "templates" / "esp32"
    metadata_path = root / "assets" / "templates" / "templates_metadata.json"

    source_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)

    source_image = source_dir / "board.png"
    template_image = template_dir / "esp32_base.png"
    cv.imwrite(str(source_image), np.zeros((900, 460, 3), dtype=np.uint8))
    cv.imwrite(str(template_image), np.zeros((20, 20), dtype=np.uint8))

    payload = {
        "source_image": "data/preparation_output_fixed/warped/board.png",
        "components": [
            {
                "component": "esp32",
                "roi_xywh": [97, 439, 281, 312],
                "files": ["assets/templates/esp32/esp32_base.png"],
            }
        ],
    }
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    bank = load_prepared_template_bank(metadata_path, rotation_turns=1)

    assert bank.canonical_size == (900, 460)
    assert bank.board_dir == source_dir.resolve()
    assert bank.component_dirs["ESP32"] == template_dir.resolve()

    x1f, y1f, x2f, y2f = bank.component_rois["ESP32"]
    assert round(x1f, 4) == 0.1656
    assert round(y1f, 4) == 0.2109
    assert round(x2f, 4) == 0.5122
    assert round(y2f, 4) == 0.8217


def test_load_prepared_template_bank_supports_generated_records_metadata(tmp_path: Path) -> None:
    root = tmp_path / "pcb_template_tools"
    source_dir = root / "data" / "preparation_output" / "warped"
    template_dir = root / "data" / "generated_templates" / "usb_port"
    metadata_path = root / "data" / "generated_templates" / "templates_metadata.json"

    source_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)

    source_image = source_dir / "board.png"
    template_image = template_dir / "usb_base.png"
    cv.imwrite(str(source_image), np.zeros((460, 900, 3), dtype=np.uint8))
    cv.imwrite(str(template_image), np.zeros((20, 20), dtype=np.uint8))

    old_clone_path = (
        "/tmp/old/clone/pcb_template_tools/data/preparation_output/warped/board.png"
    )
    payload = {
        "reference_image": old_clone_path,
        "records": [
            {
                "component": "usb_port",
                "roi_xywh": [718, 86, 124, 137],
                "files": [
                    "/tmp/old/clone/pcb_template_tools/data/generated_templates/usb_port/usb_base.png"
                ],
            }
        ],
    }
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    bank = load_prepared_template_bank(metadata_path, rotation_turns=0)

    assert bank.canonical_size == (900, 460)
    assert bank.source_image == source_image.resolve()
    assert bank.board_dir == source_dir.resolve()
    assert bank.component_dirs["USB_PORT"] == template_dir.resolve()

    x1f, y1f, x2f, y2f = bank.component_rois["USB_PORT"]
    assert round(x1f, 4) == 0.7978
    assert round(y1f, 4) == 0.1870
    assert round(x2f, 4) == 0.9356
    assert round(y2f, 4) == 0.4848


def test_load_prepared_template_bank_resolves_project_prefixed_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "pcb_template_tools"
    source_dir = root / "data" / "preparation_output_pi" / "warped"
    template_dir = root / "data" / "generated_templates_pi" / "reset_button"
    metadata_path = root / "data" / "generated_templates_pi" / "templates_metadata.json"

    source_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)

    source_image = source_dir / "board.png"
    template_image = template_dir / "reset_button_base.png"
    cv.imwrite(str(source_image), np.zeros((460, 900, 3), dtype=np.uint8))
    cv.imwrite(str(template_image), np.zeros((20, 20), dtype=np.uint8))

    payload = {
        "reference_image": "pcb_template_tools/data/preparation_output_pi/warped/board.png",
        "records": [
            {
                "component": "reset_button",
                "roi_xywh": [578, 138, 64, 91],
                "files": [
                    "pcb_template_tools/data/generated_templates_pi/reset_button/reset_button_base.png"
                ],
            }
        ],
    }
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    bank = load_prepared_template_bank(metadata_path, rotation_turns=0)

    assert bank.source_image == source_image.resolve()
    assert bank.component_dirs["RESET_BUTTON"] == template_dir.resolve()
