import argparse, csv, math, os
import cv2
import numpy as np

try:
    import librosa  # optional audio boost
except Exception:
    librosa = None

def hsv_mask_team(frame_bgr, low=(105, 70, 20), high=(130, 255, 160)):
    """Return binary mask for NAVY kit in HSV (OpenCV: H 0..180)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    low  = np.array(low, dtype=np.uint8)
    high = np.array(high, dtype=np.uint8)
    m = cv2.inRange(hsv, low, high)
    return m

TEAM_LOW = (105, 70, 20)
TEAM_HIGH = (130, 255, 160)

def read_candidates(csv_path):
    rows = []
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        r = csv.DictReader(f)
        if r.fieldnames:
            r.fieldnames = [fn.strip().lower().lstrip('\ufeff') for fn in r.fieldnames]
        for idx, row in enumerate(r, start=1):
            # tolerate columns named start/end or t0/t1
            s_val = row.get('start', row.get('t0'))
            e_val = row.get('end',   row.get('t1'))
            try:
                start = float(str(s_val).replace(',', '.'))
                end   = float(str(e_val).replace(',', '.'))
            except Exception:
                print(f"[filter_by_motion] Skipping row {idx}: cannot parse start/end -> {row}")
                continue

            score_val = row.get('score', 0.0)
            try:
                score = float(str(score_val).replace(',', '.'))
            except Exception:
                score = 0.0

            rows.append({**row, "start": start, "end": end, "score": score})
    return rows

def consecutive_true(mask, min_len):
    if min_len <= 1: return bool(mask.any())
    count = 0
    for v in mask:
        if v: 
            count += 1
            if count >= min_len: return True
        else:
            count = 0
    return False

def center_lane_mask(h, w, inner_ratio=0.60):
    # 1 in center band (inner_ratio width), 0 on sidelines
    mask = np.zeros((h, w), np.float32)
    x0 = int((1.0 - inner_ratio) * 0.5 * w)
    x1 = int(w - x0)
    mask[:, x0:x1] = 1.0
    return mask

def sample_indices(n, step):
    return list(range(0, n, max(1, step)))

def action_metrics(cap, start_s, end_s, downsample=2, step_frames=3, min_frames=24):
    """Compute per-window optical flow & motion concentration metrics."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    start_f = max(0, int(start_s * fps))
    end_f   = max(start_f+1, int(end_s * fps))
    total   = end_f - start_f
    if total < min_frames:  # too short: still evaluate, but note it
        min_frames = total

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames = []
    frames_color = []
    for i in range(total):
        ok, frame = cap.read()
        if not ok: break
        if i % step_frames: continue
        frames_color.append(frame.copy())
        if downsample > 1:
            frame = cv2.resize(frame, None, fx=1.0/downsample, fy=1.0/downsample, interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if len(frames) < 2:
        return dict(mean_flow=0, center_ratio=0, contig_ok=False, ball_speed=0, samples=len(frames), team_presence=0.0)

    h, w = frames[0].shape[:2]
    lane = center_lane_mask(h, w, inner_ratio=0.60)

    flows = []
    center_energy = []
    ball_path = []  # crude proxy: brightest small blob motion near center
    team_presence_steps = []
    prev = frames[0]
    for f in frames[1:]:
        flow = cv2.calcOpticalFlowFarneback(prev, f, None,
                                            pyr_scale=0.5, levels=1, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
        mag = cv2.magnitude(flow[...,0], flow[...,1])
        flows.append(mag)

        if frames_color:
            color_idx = min(len(flows), len(frames_color) - 1)
            team_m = hsv_mask_team(frames_color[color_idx], low=TEAM_LOW, high=TEAM_HIGH)
        else:
            team_m = np.zeros_like(mag, dtype=np.uint8)
        if downsample > 1:
            team_m = cv2.resize(team_m, (mag.shape[1], mag.shape[0]), interpolation=cv2.INTER_NEAREST)

        # motion mask
        msk = (mag > np.percentile(mag, 85))  # top motion as proxy
        # center concentration
        ctr = (msk * (lane>0)).sum() / (msk.sum()+1e-6)
        center_energy.append(ctr)

        motion_mask = msk.astype(np.uint8)
        motion_pixels = motion_mask.sum() + 1e-6
        team_pixels = ((motion_mask > 0) & (team_m > 0)).sum()
        team_presence_step = float(team_pixels / motion_pixels)
        team_presence_steps.append(team_presence_step)

        # crude ball proxy: brightest small area (top 0.1%) centroid
        thresh = np.percentile(mag, 99.9)
        k = (mag >= thresh).astype(np.uint8)
        ys, xs = np.where(k)
        if len(xs) > 0:
            ball_path.append((float(xs.mean()), float(ys.mean())))
        prev = f

    mean_flow = float(np.mean([np.mean(m) for m in flows])) if flows else 0.0
    center_ratio = float(np.mean(center_energy)) if center_energy else 0.0

    # continuity: require N consecutive frames above both mean & center thresholds
    over = []
    f_thresh = max(0.5*np.mean([np.mean(m) for m in flows]), 0.35*mean_flow)
    c_thresh = max(0.5*center_ratio, 0.25)
    for m, c in zip(flows, center_energy):
        over.append((np.mean(m) > f_thresh) and (c > c_thresh))
    contig_ok = consecutive_true(over, min_len=10)  # ~10 sampled frames in a row

    # ball speed estimate (px per sampled frame)
    ball_speed = 0.0
    if len(ball_path) >= 2:
        dists = [math.hypot(ball_path[i+1][0]-ball_path[i][0], ball_path[i+1][1]-ball_path[i][1])
                 for i in range(len(ball_path)-1)]
        ball_speed = float(np.median(dists))

    team_presence = float(np.mean(team_presence_steps)) if team_presence_steps else 0.0

    return dict(mean_flow=mean_flow, center_ratio=center_ratio, contig_ok=contig_ok,
                ball_speed=ball_speed, samples=len(frames), team_presence=team_presence)

def maybe_audio_boost(video_path, start_s, end_s):
    if librosa is None: return 0.0
    try:
        y, sr = librosa.load(video_path, sr=None, mono=True, offset=max(0.0, start_s), duration=max(0.05, end_s-start_s))
        # spectral flux as â€œcheer/excitementâ€ proxy
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        flux = np.mean(np.maximum(S[:,1:] - S[:,:-1], 0.0))
        return float(flux)
    except Exception:
        return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--csv",   required=True)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--min-motion", type=float, default=0.16, help="baseline flow magnitude gate")
    ap.add_argument("--min-green",  type=float, default=0.32, help="(legacy, ignored here if no field mask)")
    ap.add_argument("--need-ball",  type=int,   default=0,    help="legacy flag; superseded by speed gate")
    ap.add_argument("--ball-hits",  type=int,   default=0)
    ap.add_argument("--min-contig-frames", type=int, default=10)
    ap.add_argument("--min-center-ratio",  type=float, default=0.35)
    ap.add_argument("--min-ball-speed",    type=float, default=0.9, help="median px/frame (sampled) for ball proxy")
    ap.add_argument(
        "--min-flow-mean",
        "--min-flow",
        dest="min_flow_mean",
        type=float,
        default=0.55,
        help="relative scalar on window mean flow (alias --min-flow)",
    )
    ap.add_argument("--audio-boost",       type=int,   default=1,    help="use spectral flux as tiebreaker")
    ap.add_argument("--team-hsv-low",  type=str, default="105,70,20")
    ap.add_argument("--team-hsv-high", type=str, default="130,255,160")
    ap.add_argument("--min-team-pres", type=float, default=0.12, help="min fraction of motion that is team color (navy)")
    ap.add_argument("--team-bias",     type=float, default=0.25, help="score bonus weight for team presence")
    args = ap.parse_args()

    def parse_triplet(s):
        a, b, c = [int(x) for x in s.split(",")]
        return (a, b, c)

    global TEAM_LOW, TEAM_HIGH
    TEAM_LOW = parse_triplet(args.team_hsv_low)
    TEAM_HIGH = parse_triplet(args.team_hsv_high)

    rows = read_candidates(args.csv)
    if not rows:
        raise SystemExit(f"[filter] no candidates in {args.csv}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"[filter] cannot open video: {args.video}")

    kept = []
    for r in rows:
        start, end = float(r['start']), float(r['end'])
        mets = action_metrics(cap, start, end)
        # normalize â€œmean_flowâ€ gate using window stats
        mean_flow = mets['mean_flow']
        contig_ok = mets['contig_ok']
        center_ratio = mets['center_ratio']
        ball_speed = mets['ball_speed']
        team_presence = mets.get('team_presence', 0.0)

        # hard gates
        g_cont = contig_ok
        g_center = (center_ratio >= args.min_center_ratio)
        g_ball   = (ball_speed >= args.min_ball_speed)
        g_team   = (team_presence >= args.min_team_pres)

        score = 0.45 * min(2.0, mean_flow) \
              + 0.30 * center_ratio \
              + 0.25 * min(2.0, ball_speed) \
              + args.team_bias * team_presence

        if args.audio_boost:
            score += 0.10 * maybe_audio_boost(args.video, start, end)

        # final decision: require continuity + center; allow either ball or strong flow; prefer team
        ok = g_cont and g_center and (g_ball or (mean_flow > 1.2*args.min_flow_mean))
        if not ok and g_team and contig_ok and center_ratio >= (args.min_center_ratio + 0.03):
            ok = True

        why = []
        if not g_cont:   why.append("no_continuity")
        if not g_center: why.append("off_center")
        if not g_ball and mean_flow <= 1.2*args.min_flow_mean: why.append("no_ball_speed")
        if ok:
            outrow = dict(r)
            outrow.update(dict(action_score=float(score),
                               mean_flow=float(mean_flow),
                               center_ratio=float(center_ratio),
                               ball_speed=float(ball_speed),
                               team_presence=float(team_presence),
                               why="keep"))
            kept.append(outrow)
        else:
            # keep a note for debugging; not written out
            pass

    # sort by our action_score if present (desc)
    if kept and 'action_score' in kept[0]:
        kept = sorted(kept, key=lambda d: d.get('action_score', 0.0), reverse=True)

    # write filtered CSV
    if kept:
        fieldnames = list(kept[0].keys())
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in kept:
                w.writerow(row)
        print(f"[filter] kept {len(kept)} clips -> {args.out}")
    else:
        print("[filter] kept 0 clips (try lowering min-ball-speed or min-center-ratio)")

    cap.release()

if __name__ == "__main__":
    main()

