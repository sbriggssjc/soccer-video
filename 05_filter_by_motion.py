import argparse, csv, cv2, numpy as np, os

def avg_green_ratio(frame_hsv):
    # HSV ranges for grassy green (tuned for daylight/turf; adjust if needed)
    lower = np.array([35, 60, 40], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(frame_hsv, lower, upper)
    return float(np.count_nonzero(mask)) / mask.size

def analyze_window(cap, fps, start_s, end_s, stride=2):
    start_f = int(max(0, start_s*fps))
    end_f   = int(end_s*fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    ret, prev = cap.read()
    if not ret: return 0.0, 0.0, 0
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motion_acc = 0.0
    green_acc  = 0.0
    ball_hits  = 0
    frames = 0

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
        motion_acc += th.mean()/255.0

        # field green %
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_acc += avg_green_ratio(hsv)

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

    if frames == 0: return 0.0, 0.0, 0
    return motion_acc/frames, green_acc/frames, ball_hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-motion", type=float, default=0.06)
    ap.add_argument("--min-green",  type=float, default=0.08)
    ap.add_argument("--need-ball",  type=int,   default=1, help="require some ball hits (1=yes,0=no)")
    ap.add_argument("--ball-hits",  type=int,   default=6, help="min frames with ball-like blob")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise SystemExit(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    kept = []
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        s = float(r["start"]); e = float(r["end"])
        motion, green, bh = analyze_window(cap, fps, s, e)
        ok = (motion >= args.min_motion) and (green >= args.min_green)
        if args.need_ball: ok = ok and (bh >= args.ball_hits)
        if ok:
            # carry forward original score but add motion bonus
            score = float(r.get("score","0"))
            new_score = 0.6*score + 0.4*min(1.0, motion*4.0)  # gentle boost for movement
            kept.append({"start": f"{s:.2f}", "end": f"{e:.2f}", "score": f"{new_score:.3f}"})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start","end","score"])
        w.writeheader(); w.writerows(kept)

if __name__ == "__main__":
    main()
