# PCB Component Detection – Board-First Reference Architecture

This version follows a **board-first detection strategy** for the FireBeetle PCB.

The current goal is:

1. **Detect the PCB board first**
2. **Warp the board into a normalized top view**
3. **Detect the ESP32 module only inside the warped board**

This architecture is intended to be a **robust reference pipeline** for the project and a clean base for later extensions such as:

- JST connector
- USB port
- Reset button

---

## Why this version is stronger

### 1) Board-first logic
Instead of searching components in the whole image, the system first tries to find the FireBeetle board itself.

This reduces:
- false positives on the background
- false detections on fingers, screen edges, and other objects
- scale variation for the ESP32 detector

### 2) Geometry + homography for board detection
The board is not treated as just another small template target.

The pipeline uses:
- contour-based board candidate generation
- quadrilateral estimation
- homography / perspective warp
- optional board verification against real board appearance

This is more suitable than direct full-frame template matching for the PCB itself.

### 3) Component detection on the warped board
After the board is normalized, the ESP32 module is searched **only inside the warped board image**.

This makes detection:
- easier to tune
- more robust against perspective changes
- easier to extend to other components later

### 4) Better engineering base for later modules
Once the board is stable, the same architecture can be reused for:
- JST
- USB
- Reset button

Each of these can later be added as:
- a dedicated detector
- a dedicated ROI on the normalized board
- a separately tuned threshold set

---

## Current pipeline

The current reference pipeline is:

1. Read frame from webcam / video / image
2. Resize frame for performance
3. Detect FireBeetle board candidate
4. Warp board to canonical view
5. Search ESP32 inside warped board
6. Draw detections on the original image

---

## Current status

### What already works reasonably well
- Board detection on many still images
- Perspective-normalized board processing
- ESP32 detection in some warped-board cases
- Faster live processing than earlier full-template board matching attempts

### What is still difficult
- Live webcam scenes with fingers and cluttered background
- Strong blur / bad focus
- Board contours mixed with internal PCB details
- Cases where the board is too small in the frame
- False board candidates from rectangular background structures

---

## Important practical observation

Image acquisition quality has a very large effect on the result.

Detection improves significantly when:
- the board is large in the frame
- the board edges are clearly visible
- fingers do not cover corners
- the background is simple
- blur is low
- lighting is soft and uniform

For this reason, a **mechanical holder / fixture** is strongly recommended.

A good setup would be:
- board supported from the back
- front side fully visible
- clean matte background
- fixed camera position
- stable distance and angle

This is expected to improve board contour detection and warp quality much more than small threshold changes alone.

---

## Recommended run commands

### 1) Folder with still images
```bash
python main.py --source images --images-dir assets/test_images --debug --wait-ms 1500 --proc-resize-width 960

python main.py --source video --video-path assets/video/Video_1.mp4 --debug --wait-ms 1 --proc-resize-width 720

python main.py --source webcam --debug --wait-ms 1 --proc-resize-width 720