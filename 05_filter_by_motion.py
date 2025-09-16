import argparse, csv, cv2, numpy as np, os


def compute_center_lane_mask(shape, inner=0.6):
    """Return a mask that highlights the central vertical lane of the frame."""

    if not shape:
        return None
    h, w = shape[:2]
    if h <= 0 or w <= 0:
        return None
    inner = max(0.0, min(1.0, float(inner)))
    inner_w = max(1, int(round(w * inner)))
    left = max(0, (w - inner_w) // 2)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[:, left:left + inner_w] = 1.0
    return mask


def has_consecutive_true(flags, need):
    """Return True if *need* consecutive truthy samples appear in *flags*."""

    if need <= 1:
        return any(flags)
    run = 0
    for val in flags:
        if val:
            run += 1
            if run >= need:
                return True
        else:
            run = 0
    return False


def snap_to_anchors(start, end, resets, pre=2.0, post=3.5):
    """Tighten a window around the closest reset anchor if available."""

    if not resets:
        return start, end

    anchor = None
    for r_start, r_end in resets:
        if start <= r_end <= end:
            anchor = r_end
            break

    if anchor is None:
        mid = 0.5 * (start + end)
        anchor = min(resets, key=lambda pair: abs(pair[1] - mid))[1]

    anchor = min(max(anchor, start), end)
    new_start = max(start, anchor - pre)
    new_end = min(end, anchor + post)
    if new_end - new_start <= 0.25:
        return start, end
    return new_start, new_end


def avg_green_ratio(frame_hsv):
    # HSV ranges for grassy green (tuned for daylight/turf; adjust if needed)
    lower = np.array([35, 60, 40], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(frame_hsv, lower, upper)
    return float(np.count_nonzero(mask)) / mask.size

def analyze_window(cap, fps, start_s, end_s, args, stride=2, center_inner=0.6):
    start_f = int(max(0, start_s*fps))
    end_f   = int(end_s*fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    ret, prev = cap.read()
    if not ret:
        return 0.0, 0.0, 0, [], []
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motion_acc = 0.0
    green_acc  = 0.0
    motion_series = []
    green_series = []
    ball_hits  = 0
    frames = 0
    center_mask = compute_center_lane_mask(prev_gray.shape, inner=center_inner)

    f = start_f+1
    while f < end_f:
        # stride read
        for _ in range(stride-1):
            cap.grab()
            f += 1
            if f >= end_f: break
        ret, frame = cap.read(); f += 1
        if not ret: break
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),(5,5),0)

        # motion
        diff = cv2.absdiff(gray, prev_gray)
        _, th = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        if center_mask is None or center_mask.shape != th.shape:
            center_mask = compute_center_lane_mask(th.shape, inner=center_inner)
        motion_norm = th.astype(np.float32) / 255.0
        motion_weighted = motion_norm * (1.0 + args.center_weight * center_mask) - args.center_weight * (1.0 - center_mask)
        frame_motion = float(np.clip(motion_weighted, 0.0, 1.0).mean())
        motion_acc += frame_motion
        motion_series.append(frame_motion)

        # field green %
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_ratio = avg_green_ratio(hsv)
        green_acc += green_ratio
        green_series.append(green_ratio)

        # crude “ball” = small bright moving blob
        bright = cv2.inRange(gray, 200, 255)
        moving_bright = cv2.bitwise_and(bright, th)
        cnts, _ = cv2.findContours(moving_bright, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if 10 <= a <= 250:  # small blob
                ball_hits += 1
                break

        prev_gray = gray
        frames += 1

    if frames == 0:
        return 0.0, 0.0, 0, motion_series, green_series
    return motion_acc/frames, green_acc/frames, ball_hits, motion_series, green_series

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-motion", type=float, default=0.16)
    ap.add_argument("--min-green",  type=float, default=0.32)
    ap.add_argument("--need-ball",  type=int,   default=1, help="Require visible ball at least once")
    ap.add_argument("--ball-hits",  type=int,   default=1, help="Min touches within window")
    ap.add_argument("--min-contig-frames", type=int, default=12,
                    help="Require N consecutive frames over motion threshold (flowing play)")
    ap.add_argument("--center-weight", type=float, default=0.25,
                    help="Downweight motion near sidelines; 0..1 extra penalty for outer bands")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise SystemExit(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    resets = []
    goal_reset_path = None
    candidate_paths = [
        os.path.join(os.path.dirname(args.csv), "goal_resets.csv"),
        os.path.join(os.path.dirname(args.out), "goal_resets.csv"),
        "goal_resets.csv",
    ]
    for cand in candidate_paths:
        if cand and os.path.exists(cand):
            goal_reset_path = cand
            break
    if goal_reset_path:
        with open(goal_reset_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    start_v = float(row.get("start", "nan"))
                    end_v = float(row.get("end", "nan"))
                except (TypeError, ValueError):
                    continue
                if not (np.isfinite(start_v) and np.isfinite(end_v)):
                    continue
                resets.append((start_v, end_v))
        resets.sort(key=lambda pair: pair[1])

    kept = []
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        s = float(r["start"]); e = float(r["end"])
        motion, green, bh, motion_series, green_series = analyze_window(cap, fps, s, e, args)
        ok = (motion >= args.min_motion) and (green >= args.min_green)
        if motion_series and green_series and args.min_contig_frames:
            over = (np.array(motion_series) > args.min_motion) & (np.array(green_series) > args.min_green)
            if not has_consecutive_true(over, args.min_contig_frames):
                ok = False
        if args.need_ball:
            ok = ok and (bh >= args.ball_hits)
        if ok:
            # carry forward original score but add motion bonus
            score = float(r.get("score","0"))
            new_score = 0.6*score + 0.4*min(1.0, motion*4.0)  # gentle boost for movement
            if resets:
                s, e = snap_to_anchors(s, e, resets)
            if e - s <= 0:
                continue
            kept.append({"start": f"{s:.2f}", "end": f"{e:.2f}", "score": f"{new_score:.3f}"})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start","end","score"])
        w.writeheader(); w.writerows(kept)

if __name__ == "__main__":
    main()
