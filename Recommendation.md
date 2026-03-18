# PCB Component Detection – Recommendation for the Team

## Summary recommendation

You can merge this branch into **`develop`**.

But the correct interpretation is:

> **experimental milestone**, not production-ready detector.

This version is useful because it gives the team:
- a working repository state,
- a clearer board-first architecture,
- improved component overlays,
- first usable detections for ESP32, JST, USB and reset button on the internal dataset.

---

## My recommendation for merging

### Merge target
- merge into **`develop`**,
- do **not** merge into `main` yet.

### Why
Because the current branch is:
- valuable for collaboration,
- useful for code review,
- good enough for internal experiments,
- but still too fragile for final real-world claims.

---

## How to describe the current state to teammates

Use this wording:

> The current detector works best on the project’s controlled images and partially on video. It is still weak on webcam and does not generalize reliably to arbitrary cropped or internet images. The implementation should be treated as a board-specific classical CV prototype.

That is technically honest and matches the logs.

---

## What the logs tell us

### Stable / useful
- Board detection is mostly available.
- JST is relatively strong.
- Reset button is often detectable.
- ESP32 often works on the internal test set.
- USB now appears, but is still more sensitive.

### Weak
- webcam remains unstable,
- generalization to unseen internet images is poor,
- hard-cropped input can break the board-first assumption,
- ROI-based logic depends on board pose and board type consistency.

---

## Why webcam is worse than video

This is expected for this pipeline.

### Webcam adds several problems at once
- lower effective detail on the board,
- autofocus hunting,
- motion blur,
- changing exposure,
- hand shake,
- pose instability,
- cluttered background.

### Video often works better because
- you likely recorded it under more stable conditions,
- the board occupies a more consistent image area,
- motion and focus are less chaotic,
- you can choose better example segments.

So it is normal that **video works while webcam is weak**.

---

## Recommendation for the course demo

### Best presentation order
1. Show **architecture**.
2. Show **controlled image results**.
3. Show **video demo**.
4. Present webcam only as an experimental mode.

### Do not claim
- robust general object detection,
- board-invariant performance,
- internet-image robustness,
- real-time robustness under arbitrary conditions.

---

## Engineering recommendation for the next iteration

### 1. Stabilize the acquisition setup first
This is the highest-value improvement.

Recommended:
- fixed camera,
- fixed distance,
- matte background,
- board holder,
- no fingers covering corners,
- diffuse lighting,
- board filling a larger image region.

This will likely improve the system more than another small threshold tweak.

### 2. Split evaluation by mode
Evaluate separately:
- image mode,
- video mode,
- webcam mode.

One average number would be misleading.

### 3. Keep the current architecture for now
Do **not** throw this away yet.

The board-first + ROI idea is still a good direction because:
- it is explainable,
- it is fast enough for a class project,
- it matches the “one fixed PCB” use case,
- it is easier to debug than a larger deep-learning system.

### 4. Add a better fallback only later
If needed later:
- ORB / feature verification,
- simple tracking / temporal smoothing,
- YOLO only if board pose and board type variation become too high.

---

## Decision recommendation

### Yes, merge to `develop`
Reason:
- your teammates can inspect and build on it,
- the branch contains meaningful progress,
- the current state is coherent enough to share.

### No, not to `main`
Reason:
- webcam is still weak,
- unseen image robustness is insufficient,
- some detections remain setup-dependent.

---

## Suggested merge note

You can add a short note like this in the PR or merge description:

> Merge board-first ROI prototype into develop. Internal test images and some video cases work reasonably. Webcam and unseen-image robustness remain limited. Further work should focus on capture standardization, dataset expansion and per-mode evaluation.

---

## Suggested commit message

```bash
feat: add board-first ROI component detection prototype
```

Alternative:

```bash
feat: improve board-first PCB detection and overlay visualization
```

---

## Final recommendation

For today, the correct endpoint is:

- cleanly document the current limitations,
- merge to `develop`,
- let teammates review,
- continue later from this milestone.

That is a strong and realistic stopping point for the day.
