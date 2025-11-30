# Ball Detector Fine-Tuning Guide

This guide describes how to collect data and fine-tune a ball-only YOLO detector that plugs into `ball_track_yolo.py`.

## 1) Collect and label data
- Target **20–30 matches** with **30–50 frames per match** (≈600–1500 annotated frames).
- Sample diverse moments: kick-offs, long passes, aerials, crosses, keeper punts, bouncing balls, volleys, corners, throw-ins, and restarts.
- Prioritize hard scenes: bright or patterned crowds, overlapping players, mid-air balls, long shadows, low sun, glare, rainy/foggy nights, turf vs. grass, overexposed sky, and camera whip pans.
- Label a **single `ball` class** with bounding boxes. Keep the box tight to the ball; avoid including player boots or nearby limbs.
- Recommended structure (Ultralytics YOLO format):
  ```
  datasets/ball/
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt   # class_id cx cy w h (normalized)
    data.yaml                  # names: ["ball"]
  ```
- Maintain a **10–20% validation split** by game to avoid leakage between train/val.

## 2) Suggested augmentations
Enable light augmentations to improve robustness without warping the ball:
- Motion blur (short, 3–5 px radius) to mimic pans or ball-in-flight streaks.
- Brightness/contrast jitter ±20%.
- Mild zoom/crop (≤10%).
- Hue/saturation jitter ±10% for different pitch colors.
- Horizontal flip **only** if your downstream pipeline is symmetric to camera side (otherwise disable flips).

Example Ultralytics config fragment (in `data.yaml` or a custom `ball.yaml`):
```yaml
names: ["ball"]
train: images/train
val: images/val
augment:
  motion: 0.2
  blur: 0.2
  hsv_h: 0.1
  hsv_s: 0.2
  hsv_v: 0.2
  scale: 0.1
  translate: 0.05
  fliplr: 0.0   # set to 0.5 if flips are acceptable
```

## 3) Train a compact YOLO model
- Start from a small backbone (e.g., `yolo11n`/`yolov8n`) for speed.
- Train until validation mAP plateaus, typically 50–100 epochs on this dataset size.
- Example command (Ultralytics >= 8.1):
  ```bash
  yolo train \
    model=yolov8n.pt \
    data=datasets/ball/data.yaml \
    epochs=80 \
    imgsz=960 \
    batch=32 \
    lr0=0.01 \
    pretrained=True \
    name=ball-detector
  ```
- Export a compact runtime weight (e.g., `yolo export model=runs/detect/ball-detector/weights/best.pt format=pt`).

## 4) Field masks and size constraints
Add simple filters to cut false positives that sit off-field or are implausibly sized:
- **Field mask:** generate a binary mask of the pitch polygon for each camera angle (see `probe_ball_masked.py` for mask handling). Ignore detections whose centroid falls outside the mask.
- **Size guard:** reject boxes whose area is unrealistically small or large for the frame size (e.g., <2 px radius or >2% of frame area). Tune thresholds per resolution.
- **Aspect guard:** balls are near-square; penalize extreme aspect ratios.

## 5) Integrate with `ball_track_yolo.py`
- Place the exported weights where the tracker can read them, e.g., `weights/ball/best.pt`.
- Run:
  ```bash
  python ball_track_yolo.py input.mp4 ball_tracks.csv weights/ball/best.pt 0.35
  ```
- The script automatically falls back to motion-based tracking if the weight path is missing, so keep weights deployed alongside your render stack.

## 6) Validation checklist
- Qualitative: run on 5–10 unseen matches; verify stable detections during long passes, headers, and goalmouth scrambles.
- Quantitative: compute precision/recall on a held-out set with the field mask applied; aim for **>0.9 precision** while keeping recall high enough for downstream trackers.
- Regression guardrails: monitor mean box size, off-field false positives, and confidence distribution before/after each retrain.

Following this process gives a ball detector that generalizes across venues and lighting and reduces “random red dot nowhere near the ball” errors permanently.
