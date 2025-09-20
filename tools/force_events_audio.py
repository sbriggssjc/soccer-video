# tools/force_events_audio.py
import argparse, numpy as np, pandas as pd, librosa
from scipy.signal import find_peaks

def smooth(x, w):
    if w <= 1: return x
    k = int(w)
    k += (k + 1) % 2  # make odd
    return np.convolve(x, np.ones(k) / k, mode="same")

def mask_from_rms(rms, t, thr_pct=65, min_run=1.2):
    thr = np.percentile(rms, thr_pct)
    on = rms >= thr
    spans = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i + 1
            while j < len(on) and on[j]:
                j += 1
            t0, t1 = t[i], t[min(j, len(t) - 1)]
            if t1 - t0 >= min_run:
                spans.append((float(t0), float(t1)))
            i = j
        else:
            i += 1
    return spans

def pick_peaks(score, t, min_gap_s=14.0, prom=None, top=None):
    hop_s = np.median(np.diff(t))
    distance = max(1, int(min_gap_s / hop_s))
    peaks, props = find_peaks(score, distance=distance, prominence=prom)
    if top:
        if "prominences" in props:
            order = np.argsort(props["prominences"])[::-1][:top]
            peaks = peaks[order]
        else:
            order = np.argsort(score[peaks])[::-1][:top]
            peaks = peaks[order]
    return t[peaks]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="CSV: forced events (GOAL_AUDIO/POST_AUDIO)")
    ap.add_argument("--mask-out", required=True, help="CSV: ball-in-play spans")
    ap.add_argument("--top-goals", type=int, default=8)
    ap.add_argument("--top-posts", type=int, default=6)
    ap.add_argument("--goal-pre", type=float, default=12.0)
    ap.add_argument("--goal-post", type=float, default=8.0)
    ap.add_argument("--post-pre", type=float, default=5.0)
    ap.add_argument("--post-post", type=float, default=4.0)
    ap.add_argument("--sr", type=int, default=None)
    args = ap.parse_args()

    y, sr = librosa.load(args.video, sr=args.sr, mono=True)
    hop = 512
    frame = 2048

    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    # Ball-in-play mask from smoothed RMS
    rms_s = smooth(rms, w=43)  # ~0.5 s
    mask = mask_from_rms(rms_s, t, thr_pct=65, min_run=1.2)
    pd.DataFrame(mask, columns=["start", "end"]).to_csv(args.mask_out, index=False)

    # GOAL candidates: high sustained energy (crowd roar)
    goalscore = smooth(rms_s, w=95)  # ~1.1 s
    goal_ts = pick_peaks(goalscore, t, min_gap_s=14.0, prom=np.percentile(goalscore, 75), top=args.top_goals)

    # POST candidates: emphasize high-frequency energy (clang)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=hop)[0]
    cen_z = (centroid - np.median(centroid)) / (np.std(centroid) + 1e-9)
    postscore = rms_s * np.clip(cen_z, 0, None)
    post_ts = pick_peaks(postscore, t, min_gap_s=10.0, prom=np.percentile(postscore, 75), top=args.top_posts)

    rows = []
    for ts in goal_ts:
        rows.append(dict(label="GOAL_AUDIO", t0=float(max(0.0, ts - args.goal_pre)), t1=float(ts + args.goal_post)))
    for ts in post_ts:
        rows.append(dict(label="POST_AUDIO", t0=float(max(0.0, ts - args.post_pre)), t1=float(ts + args.post_post)))

    if rows:
        df = pd.DataFrame(rows).sort_values("t0")
    else:
        df = pd.DataFrame(columns=["label","t0","t1"])
    df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
