#!/usr/bin/env python
import argparse
import json
import cv2
import numpy as np

print("[DEBUG] RUNNING FILE:", __file__)

def load_action_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def build_frame_index(rows):
    by_frame = {}
    for r in rows:
        f_idx = int(r.get("f", 0))
        by_frame[f_idx] = r
    return by_frame

def extract_action_point(row):
    if row is None:
        return None

    priority_keys = [
        ("action_center_x", "action_center_y"),
        ("action_x", "action_y"),
        ("ball_center_x", "ball_center_y"),
        ("ball_x", "ball_y"),
    ]

    for kx, ky in priority_keys:
        vx = row.get(kx)
        vy = row.get(ky)
        if vx is not None and vy is not None:
            return float(vx), float(vy)

    obj_ball_center = row.get("obj_ball_center")
    if isinstance(obj_ball_center, dict):
        vx = obj_ball_center.get("x")
        vy = obj_ball_center.get("y")
        if vx is not None and vy is not None:
            return float(vx), float(vy)

    ball = row.get("ball")
    if isinstance(ball, dict):
        vx = ball.get("x")
        vy = ball.get("y")
        if vx is not None and vy is not None:
            return float(vx), float(vy)

    return None

def build_action_points(by_frame, min_conf):
    if not by_frame:
        return []

    max_frame = max(by_frame.keys())
    points = []
    last_valid = None

    for i in range(max_frame + 1):
        row = by_frame.get(i)
        ax = ay = None
        if row is not None:
            is_valid = bool(row.get("is_valid", False))
            conf = float(row.get("confidence", 0.0))
            if is_valid and conf >= min_conf:
                coords = extract_action_point(row)
                if coords is not None:
                    ax, ay = coords

        if (ax is None or ay is None) and last_valid is not None:
            ax, ay = last_valid

        if ax is not None and ay is not None:
            last_valid = (ax, ay)
            points.append(last_valid)
        else:
            points.append(None)

        print(f"[TELEMETRY] frame={i} ax={ax} ay={ay}")

    return points

def main():
    ap = argparse.ArgumentParser(description="Portrait follow using action telemetry")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--telemetry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--min-conf", type=float, default=0.30)
    ap.add_argument("--debug-overlay", action="store_true")
    args = ap.parse_args()

    rows = load_action_rows(args.telemetry)
    if not rows:
        raise SystemExit(f"No telemetry rows found in {args.telemetry}")
    print("[TELEMETRY] first row:", rows[0])
    by_frame = build_frame_index(rows)
    telemetry_points = build_action_points(by_frame, args.min_conf)

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"Could not open clip: {args.clip}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w = args.width
    target_h = args.height

    # Match height
    scale = target_h / float(src_h)
    scaled_w = int(round(src_w * scale))
    scaled_h = target_h

    print(f"[ACTION-FOLLOW] clip={args.clip}, src={src_w}x{src_h}, fps={fps:.3f}")
    print(f"[ACTION-FOLLOW] scaled={scaled_w}x{scaled_h}, out={target_w}x{target_h}, scale={scale:.4f}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (target_w, target_h))
    if not writer.isOpened():
        raise SystemExit(f"Could not open VideoWriter for {args.out}")

    frame_idx = 0
    last_valid_scaled = None
    last_valid_raw = None
    half_w = target_w // 2
    max_x0 = max(0, scaled_w - target_w)

    # ========== FRAME LOOP (clean, guaranteed-safe) ==========
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Telemetry lookup
        ax = ay = None
        if frame_idx < len(telemetry_points):
            point = telemetry_points[frame_idx]
            if point is not None:
                ax, ay = point
        elif telemetry_points:
            point = telemetry_points[-1]
            if point is not None:
                ax, ay = point

        if (ax is None or ay is None) and last_valid_raw is not None:
            ax, ay = last_valid_raw

        print(f"[TELEMETRY] frame={frame_idx} ax={ax} ay={ay}")
        use_point = False

        if ax is not None and ay is not None:
            last_valid_raw = (ax, ay)
            ax_s = ax * scale
            ay_s = ay  # DO NOT scale vertical position in portrait mode
            last_valid_scaled = (ax_s, ay_s)
            use_point = True

        # Choose center point
        if not use_point:
            if last_valid_scaled is None:
                center_x = scaled_w / 2.0
                center_y = scaled_h / 2.0
            else:
                center_x, center_y = last_valid_scaled
        else:
            center_x, center_y = last_valid_scaled

        # Compute crop window
        x0 = int(round(center_x - half_w))
        x0 = max(0, min(x0, max_x0))
        x1 = x0 + target_w

        # Resize source frame to tall canvas
        resized = cv2.resize(frame, (scaled_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Crop horizontally
        crop = resized[:, x0:x1]

        # ENFORCE EXACT SIZE (prevents all OpenCV errors)
        h, w = crop.shape[:2]
        if h != target_h or w != target_w:
            crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Debug overlay
        if args.debug_overlay and last_valid_scaled is not None:
            ax_s, ay_s = last_valid_scaled
            bx = int(round(ax_s - x0))
            by = int(round(ay_s))
            bx = max(0, min(target_w - 1, bx))
            by = max(0, min(target_h - 1, by))
            cv2.circle(crop, (bx, by), 12, (0, 0, 255), -1)
            cv2.putText(crop, f"f={frame_idx}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 0), 2, cv2.LINE_AA)

        # ---- ENSURE crop is 3-channel uint8 and contiguous ----
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        elif crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)

        crop = crop.astype("uint8", copy=False)
        crop = np.ascontiguousarray(crop)

        print("CROP SHAPE:", crop.shape, crop.dtype)
        writer.write(crop)
        frame_idx += 1
    # =========================================================

    cap.release()
    writer.release()
    print(f"[ACTION-FOLLOW] wrote portrait follow clip to: {args.out}")

if __name__ == "__main__":
    main()
