# 09_split_inplay.py  (fixed)
import argparse, csv, math, re, os
from typing import List, Tuple

def fnum(x):
    if x is None: return math.nan
    s = re.sub(r"[^0-9.\-]", ".", str(x))
    try: return float(s)
    except: return math.nan

def read_intervals(path) -> List[Tuple[float,float]]:
    rows=[]
    with open(path, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            s=fnum(r.get('start')); e=fnum(r.get('end'))
            if not math.isnan(s) and not math.isnan(e) and e>s:
                rows.append((s,e))
    rows.sort(key=lambda t:t[0])
    return rows

def read_points(path) -> List[float]:
    pts=[]
    with open(path, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            t = fnum(r.get('start') or r.get('time') or r.get('t'))
            if not math.isnan(t):
                pts.append(t)
    return sorted(pts)

def merge_close(iv: List[Tuple[float,float]], max_gap: float) -> List[Tuple[float,float]]:
    if not iv: return []
    out=[]; cs,ce=iv[0]
    for s,e in iv[1:]:
        if s <= ce + max_gap: ce=max(ce,e)
        else: out.append((cs,ce)); cs,ce=s,e
    out.append((cs,ce))
    return out

def subtract_dead(seg: Tuple[float,float], dead: List[Tuple[float,float]]):
    s0,e0 = seg
    keep=[(s0,e0)]
    for ds,de in dead:
        nxt=[]
        for s,e in keep:
            if de<=s or ds>=e: nxt.append((s,e))
            else:
                if ds> s: nxt.append((s, min(ds,e)))
                if de< e: nxt.append((max(de,s), e))
        keep=nxt
        if not keep: break
    return [(s,e) for s,e in keep if e-s>1e-6]

def union(iv: List[Tuple[float,float]]):
    if not iv: return []
    iv=sorted(iv)
    out=[]; cs,ce=iv[0]
    for s,e in iv[1:]:
        if s<=ce: ce=max(ce,e)
        else: out.append((cs,ce)); cs,ce=s,e
    out.append((cs,ce))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--filtered", required=True)
    ap.add_argument("--resets", required=False, default=None)
    ap.add_argument("--ignore-before", type=float, default=15.0)
    ap.add_argument("--max-gap", type=float, default=2.0)
    ap.add_argument("--dead-before", type=float, default=1.5)
    ap.add_argument("--dead-after", type=float, default=2.5)
    ap.add_argument("--min-play", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    base = read_intervals(args.filtered)
    base = [(max(s, args.ignore-before if False else args.ignore_before), e) for s,e in base if e>args.ignore_before]  # ignore kickoff noise
    base = merge_close(base, args.max_gap)

    dead=[]
    if args.resets and os.path.exists(args.resets):
        # accept either points CSV (start/time) or interval CSV
        pts = read_points(args.resets)
        if not pts:
            iv = read_intervals(args.resets)
            pts = [s for s,_ in iv]
        for t in pts:
            dead.append((t - args.dead_before, t + args.dead_after))
        dead = [(max(0.0,s), max(0.0,e)) for s,e in dead]
        dead = union(dead)

    plays=[]
    for seg in base:
        pieces = subtract_dead(seg, dead) if dead else [seg]
        for s,e in pieces:
            if e-s >= args.min_play:
                plays.append((s,e))
    plays = merge_close(plays, args.max_gap)

    with open(args.out, "w", newline="", encoding="ascii") as fh:
        w=csv.writer(fh); w.writerow(["start","end"])
        for s,e in plays: w.writerow([f"{s:.2f}", f"{e:.2f}"])

if __name__=="__main__":
    main()
