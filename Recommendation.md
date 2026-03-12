# PCB Component Detection – Recommended V2 Structure

This version is optimized for a **robust ESP32-on-FireBeetle detection pipeline** with a realistic path to **30 FPS on webcam/video**.

## Why this version is stronger

1. **Board-first logic**
   - The PCB is localized first.
   - The board is warped into a canonical top view.
   - Component detection runs on the warped board instead of the whole camera frame.

2. **ROI-ready architecture**
   - Each component can later use a fixed ROI on the normalized board.
   - This is the biggest practical speedup for real-time detection.

3. **Hybrid detector strategy**
   - Fast primary detector: multi-scale template matching.
   - Robust fallback: ORB feature matching.
   - Temporal smoothing removes unstable one-frame false positives.

4. **Webcam-friendly settings**
   - Resize before processing.
   - MJPG + low buffer size for webcam.
   - Logging throttled to avoid performance loss.

## Recommended run commands

### 1) Folder with still images
```bash
python main.py --source images --images-dir assets/test_images --debug --wait-ms 1500 --proc-resize-width 960
```

### 2) Single image
```bash
python main.py --source image --image-path assets/test_images/Board_1.png --debug --wait-ms 0 --proc-resize-width 960
```

### 3) Video demo
```bash
python main.py --source video --video-path assets/video/Video_1.mp4 --debug --video-resize-width 960 --proc-resize-width 960 --matcher-profile balanced
```

### 4) Fast webcam demo
```bash
python main.py --source webcam --camera-index 0 --width 1280 --height 720 --camera-fps 30 --debug --proc-resize-width 960 --matcher-profile fast
```

### 5) Headless benchmark
```bash
python main.py --source video --video-path assets/video/Video_1.mp4 --headless --max-frames 300 --proc-resize-width 960 --matcher-profile fast
```

## Recommended roadmap

### Phase A – Make ESP32 reliable
- Calibrate the board warp.
- Measure template scores on your current images/videos.
- Define a board ROI for the ESP32 region.
- Tune thresholds on real webcam data.

### Phase B – Add JST / USB / Reset button
- Add one template folder per component.
- Add a dedicated ROI per component.
- Tune each component independently.
- Extend tests with positive + negative examples.

### Phase C – Make it measurable
- Add image-level metrics: precision, recall, false positives.
- Create a deterministic evaluation script for the test set.
- Save debug outputs for every failure case.

## Core engineering recommendation

For your project, I would **not** jump directly to YOLO first.

I would use this progression:
1. **Board warp + ROI-based classical CV**
2. **Hybrid template + ORB fallback**
3. **Only later YOLO if board and camera conditions vary too much**

Why?
- You already have template data and a classical CV structure.
- For one fixed board, classical CV is often simpler, faster, and easier to explain in a Bildverarbeitung course.
- 30 FPS is much more realistic once detection is restricted to warped ROIs.
