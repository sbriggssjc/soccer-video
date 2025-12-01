import argparse
import json
import cv2


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


def main():
    ap = argparse.ArgumentParser(description="Portrait follow using action_x/action_y telemetry")
    ap.add_argument("--clip", required=True, help="Input 16:9 clip (e.g., atomic)")
    ap.add_argument("--telemetry", required=True, help="action.jsonl from build_action_telemetry.py")
    ap.add_argument("--out", required=True, help="Output portrait mp4")
    ap.add_argument("--width", type=int, default=1080, help="Output width (portrait)")
    ap.add_argument("--height", type=int, default=1920, help="Output height (portrait)")
    ap.add_argument("--min-conf", type=float, default=0.30, help="Min confidence to use action point")
    ap.add_argument("--debug-overlay", action="store_true", help="Draw dot inside portrait crop")
    args = ap.parse_args()

    rows = load_action_rows(args.telemetry)
    if not rows:
        raise SystemExit(f"No telemetry rows found in {args.telemetry}")
    by_frame = build_frame_index(rows)

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"Could not open clip: {args.clip}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale source to target portrait height, keep aspect ratio
    target_w = int(args.width)
    target_h = int(args.height)
    scale = target_h / float(src_h)
    scaled_w = int(round(src_w * scale))
    scaled_h = target_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (target_w, target_h))
    if not writer.isOpened():
        raise SystemExit(f"Could not open VideoWriter for {args.out}")

    print(f"[ACTION-FOLLOW] clip={args.clip}, src={src_w}x{src_h}, fps={fps:.3f}")
    print(f"[ACTION-FOLLOW] scaled={scaled_w}x{scaled_h}, out={target_w}x{target_h}, scale={scale:.4f}")

    frame_idx = 0
    last_valid_scaled = None  # (x,y) in scaled space

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Retrieve telemetry row ---
        row = by_frame.get(frame_idx)
        use_point = False

        if row is not None:
            is_valid = bool(row.get("is_valid", False))
            conf = float(row.get("confidence", 0.0))
            if is_valid and conf >= args.min_conf:
                ax = float(row["action_x"])
                ay = float(row["action_y"])
                ax_s = ax * scale
                ay_s = ay * scale
                last_valid_scaled = (ax_s, ay_s)
                use_point = True

        # --- Choose center point ---
        if not use_point:
            if last_valid_scaled is None:
                center_x = scaled_w / 2.0
                center_y = scaled_h / 2.0
            else:
                center_x, center_y = last_valid_scaled
        else:
            center_x, center_y = last_valid_scaled

        # --- Compute crop window ---
        x0 = int(round(center_x - (target_w / 2)))
        x0 = max(0, min(x0, scaled_w - target_w))
        x1 = x0 + target_w

        # --- Resize frame to tall canvas ---
        resized = cv2.resize(frame, (scaled_w, target_h), interpolation=cv2.INTER_LINEAR)

        # --- Crop horizontal window ---
        crop = resized[:, x0:x1]

        # --- SAFETY: enforce exact output size ---
        h, w = crop.shape[:2]
        if h != target_h or w != target_w:
            crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # --- Debug overlay ---
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

        writer.write(crop)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[ACTION-FOLLOW] wrote portrait follow clip to: {args.out}")


if __name__ == "__main__":
    main()
