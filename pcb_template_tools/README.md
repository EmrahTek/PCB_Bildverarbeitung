# warp_and_rank_boards.py

Warp raw iPhone PCB photos into one fixed canonical board view and rank them by quality.

## Purpose

This script is the first step of the PCB template preparation workflow.

It takes raw board photos, detects the PCB on a white A4 sheet, warps the board
into a fixed top-down reference view, normalizes orientation, computes quality
metrics, and ranks all successful results.

This is **not** a generic PCB detector for arbitrary boards.  
It is tuned for the current project board family.

---

## Canonical board convention

All warped outputs follow one fixed convention:

- output size: `900 x 460`
- board long edge is horizontal
- **ESP32 / BLE module is on the LEFT**
- **USB + JST are on the RIGHT**

This is the key rule for stable template generation.

It does **not** matter whether the raw phone image was captured portrait or landscape.
What matters is that the **warped result** always ends up in the same canonical orientation.

---

## Recommended dataset strategy

Use three separate datasets:

### 1. Phone Template Set
Use iPhone images for clean template preparation.

- controlled lighting
- white or gray clean background
- board fully visible
- no digital zoom
- no portrait mode
- low reflection
- stable focus

Recommended size:

- `24–30` raw full-board images
- after ranking, keep `8–12` good warped boards as board references
- use the **best 3–5 warped boards** for component template extraction

### 2. Webcam Eval Set
Use webcam images only for evaluation and tuning.

Do **not** build the main template bank from webcam images first.

### 3. Pi Camera Eval Set
Use Raspberry Pi camera later as a separate domain test set.

Only add a small Pi-specific supplemental template bank if needed.

---

## Input assumptions

The detection logic works best if:

- the PCB is placed on white A4 paper
- the board is fully visible
- the board is not heavily occluded
- reflections are limited
- perspective distortion is moderate
- the background is not cluttered

---

## Project paths

Example project root:

```text
/home/emrahtek/Schreibtisch/CodeLab/Python_tutorial/Bildverarbeitung/pcb_template_tools