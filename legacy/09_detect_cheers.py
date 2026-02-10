# 09_detect_cheers.py
# Usage: python -u 09_detect_cheers.py --video out/full_game_stabilized.mp4 --out out/cheers.csv
import argparse, csv, math, numpy as np, librosa

def z(x): 
    x = np.asarray(x, float)
    m, s = np.nanmean(x), np.nanstd(x) + 1e-9
    return (x - m) / s

def merge_spans(spans, join_sec=3.0):
    out=[]
    for s,e,score in sorted(spans):
        if not out: out.append([s,e,score]); continue
        ps,pe,pc = out[-1]
        if s <= pe + join_sec:
            out[-1][1] = max(pe, e)
            out[-1][2] = max(pc, score)
        else:
            out.append([s,e,score])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--hop",   type=int, default=512)
    ap.add_argument("--sr",    type=int, default=None)
    ap.add_argument("--pre",   type=float, default=3.0)  # context before peak
    ap.add_argument("--post",  type=float, default=2.0)  # after peak
    ap.add_argument("--thresh",type=float, default=2.5)  # z-score threshold
    args = ap.parse_args()

    # audio
    y, sr = librosa.load(args.video, sr=args.sr, mono=True)
    hop = args.hop

    # onset envelope (spiky cheers)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset = z(onset)

    # high-band energy (broadband cheering / claps)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hi_band = S[(freqs>=1500)&(freqs<=6000), :]
    mid_band = S[(freqs>=500) &(freqs<=1500), :]
    hi_env = z(hi_band.mean(axis=0))
    mid_env = z(mid_band.mean(axis=0))

    # combined cheer score (favor spikes + high freq)
    score = 0.55*onset + 0.30*hi_env + 0.15*mid_env

    # smooth a touch
    win = 7
    k = np.ones(win)/win
    smooth = np.convolve(score, k, mode="same")

    # threshold and pick contiguous peaks
    t = np.nanmean(smooth) + args.thresh*np.nanstd(smooth)
    mask = smooth > t

    spans=[]
    i=0
    while i < len(mask):
        if mask[i]:
            j=i
            while j < len(mask) and mask[j]:
                j+=1
            seg = score[i:j]
            if seg.size:
                peak_idx = int(np.argmax(seg)) + i
                peak_t   = librosa.frames_to_time(peak_idx, sr=sr, hop_length=hop)
                s = max(0.0, peak_t - args.pre)
                e = peak_t + args.post
                spans.append((s,e,float(score[peak_idx])))
            i=j
        else:
            i+=1

    # merge close spans
    spans = merge_spans(spans, join_sec=1.5)
    # write csv
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start","end","score"])
        for s,e,sc in spans:
            w.writerow([f"{s:.2f}", f"{e:.2f}", f"{sc:.3f}"])

if __name__ == "__main__":
    main()
