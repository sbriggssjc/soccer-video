# 10_rank_highlights.py
import argparse, csv, math, os, subprocess, re
from typing import List, Dict, Tuple

def fnum(x):
    s = re.sub(r"[^0-9.\-]", ".", str(x))
    try: return float(s)
    except: return float("nan")

def read_csv(path)->List[Dict]:
    rows=[]
    with open(path, newline='', encoding='utf-8') as fh:
        r=csv.DictReader(fh)
        for row in r: rows.append({k.lower():v for k,v in row.items()})
    return rows

def overlaps(a:Tuple[float,float], b:Tuple[float,float], min_overlap=0.0):
    s=max(a[0],b[0]); e=min(a[1],b[1]); return max(0.0, e-s)

def robust_metric(row:Dict, candidates:List[str], default=0.0):
    for k in candidates:
        if k in row and row[k]!="":
            v=fnum(row[k])
            if not math.isnan(v): return v
    return default

def aggregate_for_segment(seg, filt_rows):
    s,e=seg; dur=max(1e-6, e-s)
    tot_ov=0.0; m_ball=0.0; m_hits=0.0; m_flow=0.0; m_team=0.0
    for r in filt_rows:
        rs=fnum(r.get("start")); re=fnum(r.get("end"))
        if math.isnan(rs) or math.isnan(re) or re<=rs: continue
        ov=overlaps((s,e),(rs,re))
        if ov<=0: continue
        tot_ov+=ov
        m_ball += ov * robust_metric(r, ["max_ball_speed","ball_speed","avg_ball_speed","balls"], 0.0)
        m_hits += ov * robust_metric(r, ["ball_hits","hits","ballhits"], 0.0)
        m_flow += ov * robust_metric(r, ["avg_flow","flow","motion","min_flow","max_flow"], 0.0)
        m_team += ov * robust_metric(r, ["team_pres","team_presence","navy_presence","avg_team_pres","min_team_pres","max_team_pres"], 0.0)
    if tot_ov<=0: tot_ov=dur
    return {
        "ball": m_ball/tot_ov,
        "hits": m_hits/tot_ov,
        "flow": m_flow/tot_ov,
        "team": m_team/tot_ov
    }

def cheer_score(seg, cheers, pre=8.0, post=2.5):
    if not cheers: return 0.0
    s,e=seg; c=(s+e)/2.0
    best=0.0
    for t in cheers:
        if c>=t-pre and c<=t+post:
            # linear inside window; stronger nearer the cheer
            dist = 0.0 if c>=t else (t-c)
            win = post if c>=t else pre
            best = max(best, 1.0 - (dist/max(1e-6,win)))
    return best

def nms_keep(segments, scores, iou_thresh=0.5):
    keep=[]
    def iou(a,b):
        s=max(a[0],b[0]); e=min(a[1],b[1])
        inter=max(0.0,e-s); ua=(a[1]-a[0])+(b[1]-b[0])-inter
        return 0.0 if ua<=0 else inter/ua
    idx=sorted(range(len(segments)), key=lambda i: scores[i], reverse=True)
    for i in idx:
        ok=True
        for j in keep:
            if iou(segments[i], segments[j])>iou_thresh:
                ok=False; break
        if ok: keep.append(i)
    return keep

def run_ffmpeg(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--plays", required=True)
    ap.add_argument("--filtered", required=True)
    ap.add_argument("--cheers", required=False)
    ap.add_argument("--video", required=True)
    ap.add_argument("--topn", type=int, default=12)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--concat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pre", type=float, default=0.0)
    ap.add_argument("--post", type=float, default=0.0)
    args=ap.parse_args()

    plays=[]
    for r in read_csv(args.plays):
        s=fnum(r.get("start")); e=fnum(r.get("end"))
        if not math.isnan(s) and not math.isnan(e) and e>s:
            plays.append([s-args.pre, e+args.post])
    plays=[(max(0.0,s),e) for s,e in plays if e-s>=0.6]

    filt = read_csv(args.filtered)
    cheers=[]
    if args.cheers and os.path.exists(args.cheers):
        for r in read_csv(args.cheers):
            t=fnum(r.get("start"))
            if not math.isnan(t): cheers.append(t)

    feats=[]; dur=[]
    for seg in plays:
        feats.append(aggregate_for_segment(seg, filt))
        dur.append(max(0.1, seg[1]-seg[0]))

    # simple robust normalization (avoid div by 0)
    eps=1e-6
    def norm(arr): 
        m=max(arr) if arr else 1.0
        return [(x/(m+eps)) for x in arr]

    balls = norm([f["ball"] for f in feats])
    hits  = norm([f["hits"] for f in feats])
    flows = norm([f["flow"] for f in feats])
    teams = norm([f["team"] for f in feats])
    cheers_s = [cheer_score(seg, cheers) for seg in plays]
    cheers_s = norm(cheers_s)
    durs = norm(dur)

    scores=[]
    for i in range(len(plays)):
        # weights tuned to prefer shots/ball speed + motion, but keep team-navy & cheer influence
        s = 2.5*balls[i] + 0.8*hits[i] + 2.0*flows[i] + 1.2*teams[i] + 1.0*cheers_s[i] + 0.3*durs[i]
        scores.append(s)

    keep_idx = nms_keep(plays, scores, iou_thresh=0.40)[:args.topn]
    chosen = sorted(keep_idx, key=lambda i: scores[i], reverse=True)

    os.makedirs(args.clips_dir, exist_ok=True)
    # render clips (accurate seek: -ss/-to after -i; re-encode to avoid empty outputs)
    for rank,i in enumerate(chosen, start=1):
        s,e=plays[i]
        dst = os.path.join(args.clips_dir, f"clip_{rank:04d}.mp4")
        run_ffmpeg([
            "ffmpeg","-hide_banner","-loglevel","warning","-y",
            "-i", args.video, "-ss", f"{s:.2f}", "-to", f"{e:.2f}",
            "-c:v","libx264","-preset","veryfast","-crf","20",
            "-c:a","aac","-b:a","160k", dst
        ])

    # concat list
    with open(args.concat,"w",encoding="ascii") as fh:
        for rank,_ in enumerate(chosen, start=1):
            path=os.path.abspath(os.path.join(args.clips_dir, f"clip_{rank:04d}.mp4")).replace("\\","\\\\")
            fh.write(f"file '{path}'\n")

    run_ffmpeg(["ffmpeg","-hide_banner","-loglevel","warning","-y",
                "-f","concat","-safe","0","-i", args.concat, "-c","copy", args.out])

if __name__=="__main__":
    main()
