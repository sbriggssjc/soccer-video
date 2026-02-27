"""Debug: Compare YOLO models on specific frames where nano found the ball."""
import cv2
import sys

video = r"D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"

# Frames where nano model found ball with decent confidence
test_frames = [95, 96, 105, 125, 126, 253, 254]

cap = cv2.VideoCapture(video)

from ultralytics import YOLO

for model_name in ["yolov8n.pt", "yolov8m.pt", "yolov8s.pt"]:
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    model = YOLO(model_name)
    names = model.names

    for fi in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(frame, verbose=False, device="cpu", conf=0.05)

        # Show ALL detections, not just sports ball
        print(f"\n  Frame {fi}:")
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            for box, conf, cls_id in zip(xyxy, confs, classes):
                cls_int = int(round(float(cls_id)))
                cls_name = names.get(cls_int, f"cls_{cls_int}")
                x0, y0, x1, y1 = map(float, box[:4])
                w = x1 - x0
                h = y1 - y0
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                print(f"    {cls_name:20s} conf={conf:.3f} cx={cx:.0f} cy={cy:.0f} w={w:.0f} h={h:.0f}")

cap.release()
print("\nDone.")
