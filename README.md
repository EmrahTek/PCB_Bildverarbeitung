# PCB Component Detection

Authors: Emrah Tekin, Elena Bühler, Ruben Straub

This project was made at **FHGR** for the **Bildverarbeitung** course. It uses
classic computer vision with **Python** and **OpenCV**. It does not use a neural
network.

The program detects important parts on a FireBeetle / ESP32 PCB. It can work
with single images, image folders, videos, webcams, IDS cameras, and a Raspberry
Pi camera.

## What This Project Does

The program looks at a camera frame or an image. Then it finds the PCB board and
some important components on it.

It detects these labels:

- `BOARD`
- `ESP32`
- `USB_PORT`
- `JST_CONNECTOR`
- `RESET_BUTTON`

The result is shown with colored boxes in an OpenCV window. The program can also
run in headless mode for tests.

## Main Idea

The project uses a simple OpenCV pipeline:

1. Read a frame from a camera, image, folder, or video.
2. Resize the frame if needed.
3. Use image filters like grayscale, blur, CLAHE, and Canny edges.
4. Find the PCB board.
5. Warp the board into one fixed view: `900 x 460`.
6. Search for components inside known board areas.
7. Use template matching to find the parts.
8. Track boxes over live frames to reduce flicker.
9. Draw the result on the image.

The board view always has the same direction:

- ESP32 / metal module on the left
- USB-C and JST on the right
- long board side horizontal

This fixed view makes template matching more stable.

## Project Structure

```text
main.py                                      Main program entry point
config/default.yaml                          Main settings for sources and detection
config/logging.yaml                          Logging settings
src/app/                                     CLI and runtime pipeline
src/camera_input/                            Webcam, IDS, Pi camera, image, and video sources
src/preprocessing/                           Filters, color helpers, and board geometry
src/detection_logic/                         Board and component detection
src/render/                                  Overlay and FPS drawing
src/utils/                                   Shared types, IO helpers, and template loading
tests/                                       Unit and smoke tests
pcb_template_tools/tools/                    Tools to build template banks
pcb_template_tools/data/generated_templates  Template bank for image/webcam/video/IDS
pcb_template_tools/data/generated_templates_pi Template bank for Raspberry Pi camera
```

## Requirements

Recommended system:

- Linux or Raspberry Pi OS
- Python 3.10 or newer
- OpenCV
- NumPy
- PyYAML
- pytest for tests

For Raspberry Pi camera support, install Picamera2 with apt. It is normally not
installed through pip.

## Installation

Clone the repository:

```bash
git clone git@github.com:EmrahTek/PCB_Bildverarbeitung.git
cd PCB_Bildverarbeitung
```

Create a virtual environment on a normal Linux/PC system:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Install test tools if needed:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
```

Check that the Python files can be imported:

```bash
.venv/bin/python -m compileall main.py src tests
```

Run the tests:

```bash
.venv/bin/python -m pytest -q
```

## Quick Start With One Image

Run detection on one test image:

```bash
.venv/bin/python main.py \
  --source image \
  --image-path pcb_template_tools/test_images/IMG_9688.JPG \
  --debug \
  --loop \
  --wait-ms 30 \
  --proc-resize-width 960
```

Press `q` in the OpenCV window to stop the program.

## Run Many Images Headless

This is useful for a quick check without a GUI:

```bash
.venv/bin/python main.py \
  --source images \
  --images-dir pcb_template_tools/test_images \
  --headless \
  --debug \
  --wait-ms 1 \
  --proc-resize-width 960
```

## Webcam

List video devices:

```bash
.venv/bin/python main.py --list-video-devices
```

Check if the webcam opens:

```bash
.venv/bin/python main.py \
  --source webcam \
  --camera-device 0 \
  --camera-backend any \
  --camera-open-check \
  --debug
```

Run live webcam detection:

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

## Raspberry Pi Camera

Install the needed apt packages on the Raspberry Pi:

```bash
sudo apt update
sudo apt install python3-picamera2 python3-opencv python3-yaml
```

Use `/usr/bin/python3` for Pi camera commands, because Picamera2 is installed as
a system package.

Check the camera and save the first frame:

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

Run live Pi camera detection:

```bash
PYTHONPATH=. /usr/bin/python3 main.py \
  --source picamera \
  --camera-index 0 \
  --debug \
  --width 1280 \
  --height 720 \
  --camera-fps 30 \
  --proc-resize-width 720
```

`--proc-resize-width 720` is a stable value for the Pi camera. Smaller values
can be faster, but small parts like `RESET_BUTTON` can become less accurate.

## IDS Camera

Open test:

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

## Video

Run detection on a video file:

```bash
.venv/bin/python main.py \
  --source video \
  --video-path pcb_template_tools/test_video/WIN_20260420_11_54_52_Pro.mp4 \
  --debug \
  --video-resize-width 720 \
  --proc-resize-width 720
```

## Template Preparation

The project uses template banks. A template bank contains small images of the
board and its components.

The workflow has two steps:

1. Warp raw board photos into the fixed board view.
2. Select component ROIs and export component templates.

Create a Pi camera template bank:

```bash
.venv/bin/python pcb_template_tools/tools/warp_and_rank_boards.py \
  --input-dir pcb_template_tools/data/raw_pi \
  --output-dir pcb_template_tools/data/preparation_output_pi

.venv/bin/python pcb_template_tools/tools/extract_templates.py \
  --report pcb_template_tools/data/preparation_output_pi/board_quality_report.json \
  --top-k 4 \
  --output-dir pcb_template_tools/data/generated_templates_pi
```

When the ROI window opens, select these parts:

1. `esp32`
2. `usb_port`
3. `jst_connector`
4. `reset_button`

Confirm a box with `ENTER` or `SPACE`. Press `c` if you want to draw the box
again.

## Important Settings

Most settings are in:

```text
config/default.yaml
```

Useful settings:

- `runtime.processing_width`: default processing width
- `source_profiles.picamera`: special settings for the Pi camera
- `templates`: paths to board and component templates
- `components.*.roi`: search areas for each component
- `tracking`: settings for live video smoothing

## Troubleshooting

If the camera does not open:

```bash
ls /dev/video*
.venv/bin/python main.py --list-video-devices
```

If Picamera2 is missing in the virtual environment, use:

```bash
PYTHONPATH=. /usr/bin/python3 main.py --source picamera --camera-open-check --debug
```

If detection flickers:

- Use stable light.
- Keep the PCB large enough in the frame.
- Keep the camera distance fixed.
- Use `--proc-resize-width 720` on the Pi camera.
- Make a new Pi template bank if the camera view changed a lot.

## License

This project is licensed under the **GNU General Public License v3.0 or later**
(`GPL-3.0-or-later`). See [LICENSE](LICENSE).
