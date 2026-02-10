import argparse, csv, math, os
import cv2
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Proxy or full video (use proxy for speed)")
    ap.add_argument("--csv", required=True, help="Input highlights.csv from detector")
    ap.add_argument("--out", default="out/highlights_tight.csv", help="Output refined CSV")
    ap.add_argument("--target", type=float, default=8.0, help="target clip length (sec)")
    ap.add_argument("--pre", type=float, default=3.0, help="seconds before peak")
    ap.add_argument("--post", type=float, default=5.0, help="seconds after peak")
    ap.add_argument("--sample_fps", type=float, default=6.0, help="sampling fps for motion scoring")
    return ap.parse_args()

def clamp(a, lo, hi): return max(lo, min(a, hi))

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # sample every N frames to be fast
    step = max(1, int(round(fps / args.sample_fps)))

    rows = []
    with open(args.csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                start = float(row["start"]); end = float(row["end"])
                score = float(row.get("score", 0))
            except Exception:
                continue
            start = clamp(start, 0, duration)
            end   = clamp(end,   0, duration)
            if end <= start: 
                continue

            start_f = int(start * fps); end_f = int(end * fps)

            # Seek and score motion
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            prev = None
            scores = []; times = []
            fidx = start_f
            while fidx <= end_f:
                ok, frame = cap.read()
                if not ok: break
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) < fidx:  # safety
                    break
                if (fidx - start_f) % step != 0:
                    fidx += 1
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5,5), 0)
                if prev is not None:
                    diff = cv2.absdiff(gray, prev)
                    scores.append(float(np.mean(diff)))
                    times.append(fidx / fps)
                prev = gray
                fidx += 1

            if not scores:
                # fallback: center of original window
                peak_t = (start + end) / 2.0
            else:
                # smooth & pick peak
                s = np.array(scores, dtype=np.float32)
                if len(s) >= 5:
                    s = cv2.blur(s.reshape(-1,1), (5,1)).reshape(-1)
                peak_t = times[int(np.argmax(s))]

            # build tight window
            tgt = max(1.0, args.target)
            pre = args.pre; post = args.post
            # keep length ~= target by scaling pre/post if needed
            length = pre + post
            if abs(length - tgt) > 0.5:
                scale = tgt / length
                pre *= scale; post *= scale

            new_start = clamp(peak_t - pre, start, end)
            new_end   = clamp(peak_t + post, new_start + 0.5, end)  # at least 0.5s

            rows.append({"start": f"{new_start:.2f}", "end": f"{new_end:.2f}", "score": f"{score:.3f}"})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start","end","score"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
