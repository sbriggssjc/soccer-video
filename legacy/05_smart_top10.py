# 05_smart_top10.py
import os, glob, csv, math
import numpy as np
import cv2

def activity_profile(path, sample_fps=6):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    stride = max(1, int(round(fps / sample_fps)))

    ok, prev = cap.read()
    if not ok:
        cap.release()
        return [], [], []
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    times = []
    cover = []  # fraction of pixels moving
    mag   = []  # mean abs diff (0..1)

    while True:
        # skip to next sampled frame
        for _ in range(stride - 1):
            if not cap.grab():
                break
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev)
        prev = gray

        m = float(diff.mean()) / 255.0
        mask = (diff > 10).astype(np.uint8)
        c = float(mask.mean())

        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        times.append(t); cover.append(c); mag.append(m)

    cap.release()
    return times, np.array(cover), np.array(mag)

def find_first_active(times, cover, mag, sustain_sec=1.25, sample_fps=6):
    if len(times) == 0:
        return 0.0
    # adaptive baseline: “idle” amount of motion/coverage
    cov_base = np.percentile(cover, 20)
    cov_thr = max(0.010, cov_base + 0.010)
    mag_thr = max(0.010, np.percentile(mag, 20) + 0.005)

    active = (cover > cov_thr) & (mag > mag_thr)
    need = max(1, int(round(sustain_sec * sample_fps)))

    run = 0
    for i, a in enumerate(active):
        run = run + 1 if a else 0
        if run >= need:
            # back off 1s for context
            return max(0.0, times[i] - 1.0)
    return 0.0

def audio_rms(path):
    try:
        import librosa
        y, sr = librosa.load(path, sr=None, mono=True)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512).ravel()
        return float(np.percentile(rms, 90))
    except Exception:
        return 0.0

def score_clip(path):
    times, cover, mag = activity_profile(path, sample_fps=6)
    inpoint = find_first_active(times, cover, mag, sustain_sec=1.25, sample_fps=6)
    motion_score = float(np.percentile(mag[cover>0] if cover.size and (cover>0).any() else mag, 80)) if len(mag) else 0.0
    a = audio_rms(path)
    score = 0.65 * motion_score + 0.35 * a
    duration = clip_duration(path)
    return {
        "path": os.path.abspath(path),
        "inpoint": round(inpoint, 3),
        "duration": duration,
        "score": score,
        "motion": motion_score,
        "audio": a
    }

def clip_duration(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(frames / fps) if frames else 0.0

def main():
    root = os.getcwd()
    cand_dirs = [os.path.join(root, "out", "clips_acc"),
                 os.path.join(root, "out", "clips")]
    files = []
    for d in cand_dirs:
        files += sorted(glob.glob(os.path.join(d, "clip_*.mp4")))
    if not files:
        print("No candidate clips found in out\\clips_acc or out\\clips")
        return

    rows = []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] scoring {os.path.basename(f)}")
        rows.append(score_clip(f))

    # prefer clips that actually get active early; drop ones where inpoint is near end
    filtered = [r for r in rows if r["duration"] - r["inpoint"] >= 6.0]  # need at least ~5s after trim
    if len(filtered) < 10:
        filtered = rows  # fall back if too aggressive

    # shorter, punchier social clips
    MAX_LEN = 18.0  # seconds

    sorted_rows = sorted(filtered, key=lambda r: r["score"], reverse=True)[:10]

    os.makedirs(os.path.join("out"), exist_ok=True)
    # CSV for visibility
    csv_path = os.path.join("out", "smart_top10.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path","inpoint","duration","motion","audio","score"])
        w.writeheader()
        for r in sorted_rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    # Concat list with in/out points
    txt_path = os.path.join("out", "smart_top10_concat.txt")
    with open(txt_path, "w", newline="\n", encoding="ascii") as f:
        for r in sorted_rows:
            outpoint = min(r["duration"], r["inpoint"] + MAX_LEN)
            f.write(f"file '{r['path']}'\n")
            f.write(f"inpoint {r['inpoint']:.3f}\n")
            f.write(f"outpoint {outpoint:.3f}\n")
    print(f"Wrote {txt_path}")

if __name__ == "__main__":
    main()
