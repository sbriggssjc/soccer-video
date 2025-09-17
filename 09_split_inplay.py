# 09_split_inplay.py
import argparse, csv, math, re
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

def merge_close(iv: List[Tuple[float,float]], max_gap: float) -> List[Tuple[float,float]]:
    if not iv: return []
    out=[]; cs,ce=iv[0]
    for s,e in iv[1:]:
        if s <= ce + max_gap: ce=max(ce,e)
        else: out.append((cs,ce)); cs,ce=s,e
    out.append((cs,ce))
    return out

def subtract_dead(seg: Tuple[float,float], dead: List[Tuple[float,float]]):
    # subtract union of dead intervals from a single segment, return list of kept pieces
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
    ap.add_argument("--ignore-before", type=float, default=15.0)   # skip pre-kickoff noise
    ap.add_argument("--max-gap", type=float, default=2.0)          # merge windows if within 2s
    ap.add_argument("--dead-before", type=float, default=1.5)      # around restarts to cut out dead time
    ap.add_argument("--dead-after", type=float, default=2.5)
    ap.add_argument("--min-play", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    base = read_intervals(args.filtered)
    base = [(max(s, args.ignore_before), e) for s,e in base if e>args.ignore_before]
    base = merge_close(base, args.max_gap)

    dead=[]
    if args.resets:
        for t,_ in read_intervals(args.resets):
            dead.append((t-args.dead-before, t+args.dead_after))
        dead = union([(max(0.0,s), max(0.0,e)) for s,e in dead])

    plays=[]
    for seg in base:
        pieces = subtract_dead(seg, dead) if dead else [seg]
        for s,e in pieces:
            if e-s >= args.min_play:
                plays.append((s,e))

    # final tidy merge (again) in case subtract created close neighbors
    plays = merge_close(plays, args.max_gap)

    with open(args.out, "w", newline="", encoding="ascii") as fh:
        cw=csv.writer(fh); cw.writerow(["start","end"])
        for s,e in plays: cw.writerow([f"{s:.2f}", f"{e:.2f}"])

if __name__=="__main__":
    main()
