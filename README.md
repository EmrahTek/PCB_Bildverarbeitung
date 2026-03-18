# PCB Component Detection – End-of-Day README

This repository contains a **classical computer vision prototype** for detecting components on a **DFRobot / FireBeetle-style PCB**.

The current implementation follows a **board-first pipeline**:

1. detect the board in the full image,
2. normalize it conceptually to a known layout,
3. search for components in **predefined ROIs**,
4. visualize detections with colored overlays.

The main target components are:
- **BOARD**
- **ESP32 module**
- **USB port**
- **JST connector**
- **Reset button**

---

## Current architecture

The current version is not a generic PCB detector.
It is a **board-specific prototype** tuned for one PCB family and one approximate layout.

### Detection strategy

- **Board detection** is done first.
- After the board is found, the detector uses **fixed ROI regions** for the components.
- Inside each ROI, the system uses **template matching** and score thresholds.
- The overlay system is optimized for readability with short labels:
  - `BRD`
  - `ESP`
  - `USB`
  - `JST`
  - `RST`

---

## What works well now

Based on today’s tests, the current version works **reasonably well on the controlled internal image set**.

### Stronger parts
- **BOARD** detection is mostly stable on the project test set.
- **JST connector** became one of the strongest component detections.
- **Reset button** is now detected in many frames.
- **ESP32** is often detected on the internal dataset.
- Overlay readability is much better than before.

### Especially improved today
- USB and JST ROIs now produce detections.
- Labels are smaller, more consistent, and less visually tiring.
- The visualization is now suitable for demo screenshots.

---

## What does **not** work reliably yet

This is important for the team:

### 1. Webcam output is still weak
The webcam result is currently **not robust enough** for a reliable final demonstration under unconstrained conditions.

Typical problems:
- motion blur,
- autofocus changes,
- exposure changes,
- low apparent board size in frame,
- hand occlusion,
- background clutter,
- unstable board pose.

### 2. Internet images / unseen images are not reliable
Today’s tests showed that the detector often works on the internal dataset, but **fails or behaves inconsistently** on:
- cropped custom images,
- screenshots,
- internet images,
- different board revisions,
- strongly rotated or perspective-heavy views.

### 3. Hard crops can break the logic
If the image is already cropped aggressively, the board detector and the ROI assumptions can become inconsistent.
This can produce strange detections or missed detections.

---

## Why it fails on cropped / internet images

The reason is not random. It comes from the current design.

### The current detector assumes:
- a specific board layout,
- a roughly similar board scale after detection,
- a similar visual appearance to the templates,
- ROIs that match the expected component positions.

If one of these changes, performance drops.

### Typical failure reasons

#### A) ROI mismatch
The component search happens in predefined board regions.
If the board is from another revision, rotated differently, cropped differently, or not normalized well enough, the ROI no longer matches the true component position.

#### B) Template specificity
Template matching is appearance-sensitive.
It depends strongly on:
- lighting,
- contrast,
- blur,
- resolution,
- board revision,
- color tone,
- scale.

#### C) Geometry dependency
The board-first logic works best when the board is clearly visible as one structured object.
When the input is already cut, partially visible, or visually very different, the first stage becomes weaker.

So the current result is consistent with the logs: **the pipeline is tuned for the project’s own data distribution, not for arbitrary web images**.

---

## Interpreting today’s logs

The logs show a clear pattern:

- **JST** often has strong scores.
- **Reset button** is also frequently above threshold.
- **USB** is more borderline and sensitive.
- **ESP32** is moderate: often detected, but not fully stable.
- **Webcam** remains the weakest mode.

This means the current system is best described as:

> **good prototype for controlled images and some video frames, but not yet a robust real-world webcam detector**.

---

## Recommended run commands

### Test images
```bash
python main.py --source images --images-dir assets/test_images --debug --wait-ms 1500 --proc-resize-width 960
```

### Video demo
```bash
python main.py --source video --video-path assets/video/Video_1.mp4 --debug --wait-ms 1 --proc-resize-width 960
```

### Webcam demo
```bash
python main.py --source webcam --debug --wait-ms 1 --proc-resize-width 720
```

For the current state, **video is more trustworthy than webcam**.

---

## Practical recommendation for demo use

For course presentation or teammate review, use this order:

1. **still images** to show the idea,
2. **video** as the practical demo,
3. **webcam only as an experimental mode**, not as the main proof.

---

## Recommendation for branch status

This branch is suitable to merge into **`develop`** as an **experimental prototype milestone**.
It should **not** be presented as a finished, robust detector.

Suggested status wording:

> Board-first classical CV prototype with ROI-based component detection. Works reasonably on controlled test images and some video frames. Webcam and unseen-image generalization are still limited.

---

## Suggested next steps

### Priority 1 – Data and setup
- build a more controlled capture setup,
- collect more images from the same board revision,
- collect dedicated templates for USB / JST / reset from your exact hardware,
- reduce blur and exposure changes.

### Priority 2 – Evaluation
- separate metrics for image / video / webcam,
- save false positives and false negatives,
- evaluate per component, not only per frame.

### Priority 3 – Algorithmic improvements
- better board normalization,
- per-component threshold tuning,
- optional ORB/feature fallback,
- optional temporal smoothing for video/webcam,
- later: YOLO only if board variability becomes too high.

---

## Honest conclusion

This is a **good engineering step forward**, not a final solution.

The prototype now demonstrates:
- a clear board-first architecture,
- functioning ROI-based component detection,
- readable visualization,
- stronger results on the internal dataset.

But it also clearly shows the current limitation:

> the method is still strongly tied to the training/setup conditions and does not yet generalize well to arbitrary webcam scenes or internet images.
