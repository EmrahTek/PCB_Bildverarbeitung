# PCB Component Detection

Authors: Emrah Tekin, Elena Bühler, Ruben Straub

This project detects the main components on a FireBeetle / ESP32-based PCB with
classic computer-vision techniques.

Detected labels:

- `BOARD`
- `ESP32`
- `USB_PORT`
- `JST_CONNECTOR`
- `RESET_BUTTON`

The main runtime target is stable live component detection on a Raspberry Pi 5
with a Sony/Pi camera. The older iPhone-based template bank is kept for default,
image, webcam, video, and IDS workflows; a separate template bank is used for
the Pi camera profile.

## Short Summary

The pipeline works as follows:

1. A frame is read from a camera or image source.
2. The large PCB board is found first.
3. The board is warped into a canonical `900 x 460` view.
4. Components are searched inside expected ROI areas on the board.
5. During live video, board pose and component boxes are tracked to reduce flicker.
6. The result is drawn on the OpenCV GUI with colored boxes.

The canonical board orientation is always the same:

- ESP32 / metal module on the left
- USB-C and JST on the right
- board long edge horizontal

## Project Folders

```text
config/default.yaml                         Main detector and source settings
main.py                                     Application entry point
src/                                        Camera, pipeline, detection, and render code
tests/                                      Unit and smoke tests
logs/app.log                                Runtime log file
pcb_template_tools/tools/warp_and_rank_boards.py
pcb_template_tools/tools/extract_templates.py
pcb_template_tools/data/pcb_iphone_raw      iPhone raw photos
pcb_template_tools/data/raw_pi              Pi camera raw photos
pcb_template_tools/data/preparation_output  iPhone warp/mask/preview outputs
pcb_template_tools/data/preparation_output_pi
pcb_template_tools/data/generated_templates
pcb_template_tools/data/generated_templates_pi
```

Template separation:

- The default, image, webcam, video, and IDS profiles use the iPhone bank:
  `pcb_template_tools/data/generated_templates`
- The `--source picamera` profile uses the Pi bank:
  `pcb_template_tools/data/generated_templates_pi`

This separation comes from the `source_profiles.picamera` section in
`config/default.yaml`.

## Installation

Project path on this machine:

```bash
cd /home/emrahtek/Schreibtisch/CodeLab/PCB_Bauteilerkennung
```

Virtual environment for a normal Linux/PC environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Quick import/syntax check:

```bash
.venv/bin/python -m compileall main.py src tests
```

Run tests:

```bash
.venv/bin/python -m pytest -q
```

Note: on Raspberry Pi, `picamera2` usually comes from apt packages. For Pi
camera commands, `/usr/bin/python3` is often more reliable than `.venv/bin/python`.

## Raspberry Pi Camera Live Run

Required apt packages on the Pi:

```bash
sudo apt update
sudo apt install python3-picamera2 python3-opencv python3-yaml
```

Test the camera and first frame:

```bash
cd /home/emrahtek/Schreibtisch/CodeLab/PCB_Bauteilerkennung

PYTHONPATH=. /usr/bin/python3 main.py \
  --source picamera \
  --camera-index 0 \
  --camera-open-check \
  --save-first-frame /tmp/picamera-first-frame.png \
  --debug \
  --width 1280 \
  --height 720 \
  --camera-fps 30
```

Main command for live Pi camera detection:

```bash
cd /home/emrahtek/Schreibtisch/CodeLab/PCB_Bauteilerkennung

PYTHONPATH=. /usr/bin/python3 main.py \
  --source picamera \
  --camera-index 0 \
  --debug \
  --width 1280 \
  --height 720 \
  --camera-fps 30 \
  --proc-resize-width 720
```

`--debug` shows score/ratio text above boxes and writes detailed logs. This is
the most stable Pi camera mode for this project; use it during calibration and
runtime checks.

When score text is not needed, remove `--debug`. Detection settings stay the
same; only the on-screen score/ratio text is hidden:

```bash
PYTHONPATH=. /usr/bin/python3 main.py \
  --source picamera \
  --camera-index 0 \
  --width 1280 \
  --height 720 \
  --camera-fps 30 \
  --proc-resize-width 720
```

Note: `--proc-resize-width 720` is the selected stable Pi camera setting. Lower
values can improve FPS, but they may reduce box precision for small components
such as `RESET_BUTTON`.

Press `q` in the GUI window to exit.

## Single Image, Folder, and Video Commands

Single-image GUI:

```bash
.venv/bin/python main.py \
  --source image \
  --image-path pcb_template_tools/test_images/IMG_9688.JPG \
  --debug \
  --loop \
  --wait-ms 30 \
  --proc-resize-width 960
```

Headless test for images in a folder:

```bash
.venv/bin/python main.py \
  --source images \
  --images-dir pcb_template_tools/test_images \
  --headless \
  --debug \
  --wait-ms 1 \
  --proc-resize-width 960
```

Show folder images one by one in the GUI:

```bash
.venv/bin/python main.py \
  --source images \
  --images-dir pcb_template_tools/test_images \
  --debug \
  --wait-ms 1500 \
  --proc-resize-width 960
```

Video GUI:

```bash
.venv/bin/python main.py \
  --source video \
  --video-path pcb_template_tools/test_video/WIN_20260420_11_54_52_Pro.mp4 \
  --debug \
  --video-resize-width 720 \
  --proc-resize-width 720
```

Fast headless video test:

```bash
.venv/bin/python main.py \
  --source video \
  --video-path pcb_template_tools/test_video/WIN_20260420_11_54_52_Pro.mp4 \
  --headless \
  --debug \
  --max-frames 80 \
  --video-resize-width 720 \
  --video-stride 2 \
  --proc-resize-width 720
```

## Webcam and IDS Commands

List video devices:

```bash
.venv/bin/python main.py --list-video-devices
```

Webcam open test:

```bash
.venv/bin/python main.py \
  --source webcam \
  --camera-device 0 \
  --camera-backend any \
  --camera-open-check \
  --debug
```

Live webcam detection:

```bash
.venv/bin/python main.py \
  --source webcam \
  --camera-device 0 \
  --camera-backend any \
  --debug \
  --width 1280 \
  --height 720 \
  --proc-resize-width 720
```

IDS camera open test:

```bash
.venv/bin/python main.py \
  --source ids \
  --camera-device /dev/video0 \
  --camera-backend auto \
  --camera-open-check \
  --debug \
  --width 1600 \
  --height 1200 \
  --disable-mjpg
```

Live IDS detection:

```bash
.venv/bin/python main.py \
  --source ids \
  --camera-device /dev/video0 \
  --camera-backend auto \
  --debug \
  --width 1600 \
  --height 1200 \
  --proc-resize-width 960 \
  --disable-mjpg
```

Use this when the IDS uEye SDK / pyueye path is required:

```bash
.venv/bin/python main.py \
  --source ids \
  --camera-device 0 \
  --camera-backend pyueye \
  --debug \
  --width 1600 \
  --height 1200 \
  --proc-resize-width 960 \
  --disable-mjpg
```

## Template Preparation Workflow

Template preparation has two steps:

1. Warp raw board photos into the canonical board view.
2. Extract the component ROI/template bank from the best warped boards.

### Regenerate the Pi Camera Template Bank

Put raw Pi photos here:

```text
pcb_template_tools/data/raw_pi
```

Warp and rank by quality:

```bash
.venv/bin/python pcb_template_tools/tools/warp_and_rank_boards.py \
  --input-dir pcb_template_tools/data/raw_pi \
  --output-dir pcb_template_tools/data/preparation_output_pi
```

Extract component templates:

```bash
.venv/bin/python pcb_template_tools/tools/extract_templates.py \
  --report pcb_template_tools/data/preparation_output_pi/board_quality_report.json \
  --top-k 4 \
  --output-dir pcb_template_tools/data/generated_templates_pi
```

When ROI windows open, select these boxes in order:

1. `esp32`
2. `usb_port`
3. `jst_connector`
4. `reset_button`

After drawing a box, confirm with `ENTER` or `SPACE`. If the box is wrong, press
`c` and select it again.

Reuse previously selected ROIs:

```bash
.venv/bin/python pcb_template_tools/tools/extract_templates.py \
  --report pcb_template_tools/data/preparation_output_pi/board_quality_report.json \
  --top-k 4 \
  --output-dir pcb_template_tools/data/generated_templates_pi \
  --roi-file pcb_template_tools/data/generated_templates_pi/component_rois.json
```

### Regenerate the iPhone Template Bank

iPhone raw photos are stored here:

```text
pcb_template_tools/data/pcb_iphone_raw
```

Warp:

```bash
.venv/bin/python pcb_template_tools/tools/warp_and_rank_boards.py \
  --input-dir pcb_template_tools/data/pcb_iphone_raw \
  --output-dir pcb_template_tools/data/preparation_output
```

Extract templates:

```bash
.venv/bin/python pcb_template_tools/tools/extract_templates.py \
  --report pcb_template_tools/data/preparation_output/board_quality_report.json \
  --top-k 3 \
  --output-dir pcb_template_tools/data/generated_templates \
  --roi-file pcb_template_tools/data/generated_templates/component_rois.json
```

## Log Checks

Show the latest log lines:

```bash
tail -n 80 logs/app.log
```

A successful live debug line typically contains these labels:

```text
labels=BOARD, ESP32, JST_CONNECTOR, RESET_BUTTON, USB_PORT
```

In Pi camera logs, `source=picamera:0` and `labels=...` lines confirm that the
runtime is using the correct source.

## Important Settings

Main configuration file:

```text
config/default.yaml
```

Frequently used settings:

- `runtime.processing_width`: default processing width.
- `source_profiles.picamera`: template, board, and component overrides for Pi camera.
- `source_profiles.picamera.templates`: paths for the Pi template bank.
- `tracking.board_bbox_pad_right`: expands only the displayed `BOARD` box on the right.
- `components.USB_PORT.output_bbox_pad_right`: expands the output USB box on the right.
- `components.JST_CONNECTOR.output_bbox_pad_right`: expands the output JST box on the right.
- `components.*.layout_anchor`: locks the component to the expected layout ROI when board and ROI evidence is reliable.
- `components.*.layout_fallback_score`: score shown when layout fallback is used.
- `components.*.search_roi_expansion`: expands the component search area.
- `components.*.layout_roi_left_trim`: trims the left side of the layout box; used to keep USB/JST boxes further right.

The latest live Pi profile settings keep the outer right edges of USB and JST
inside their boxes. This improves the visual overlay without aggressively
changing the template-matching logic.

## Troubleshooting

If Picamera2 is not available inside the virtual environment:

```bash
PYTHONPATH=. /usr/bin/python3 main.py --source picamera --camera-open-check --debug
```

If the camera does not open:

```bash
ls /dev/video*
.venv/bin/python main.py --list-video-devices
```

Save the first Pi camera frame:

```bash
PYTHONPATH=. /usr/bin/python3 main.py \
  --source picamera \
  --camera-index 0 \
  --camera-open-check \
  --save-first-frame /tmp/picamera-first-frame.png \
  --debug \
  --width 1280 \
  --height 720 \
  --camera-fps 30
```

If detection flickers:

- Use steadier lighting.
- Keep the board centered and reasonably large in the frame.
- Keep the camera-to-board distance fixed.
- Use `--proc-resize-width 720` for Pi camera; lower values may hurt the small
  `RESET_BUTTON` box.
- Capture new raw Pi photos and regenerate the `generated_templates_pi` bank.

## Developer Check Commands

Syntax/import check:

```bash
python3 -m compileall main.py src tests
```

If pytest is installed:

```bash
python3 -m pytest -q
```

Short headless image check:

```bash
.venv/bin/python main.py \
  --source images \
  --images-dir pcb_template_tools/test_images \
  --headless \
  --debug \
  --wait-ms 1 \
  --proc-resize-width 960
```
