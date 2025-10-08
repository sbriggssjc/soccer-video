import json
import math
import sys
import statistics as stats
from pathlib import Path

p = Path(r"out\render_logs\tester_022__SHOT.ball.jsonl")
if not p.exists():
    print("no ball path file:", p)
    sys.exit(2)

bx, by = [], []
for i, line in enumerate(p.open("r", encoding="utf-8")):
    d = json.loads(line)
    bx.append(float(d.get("bx", 0)))
    by.append(float(d.get("by", 0)))


def diffs(a):
    return [a[i + 1] - a[i] for i in range(len(a) - 1)]


vx = diffs(bx)
vy = diffs(by)
spd = [math.hypot(vx[i], vy[i]) for i in range(len(vx))]
if not spd:
    print("empty speed")
    sys.exit(2)


def pct(a, q):
    k = (len(a) - 1) * q / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted(a)[int(k)]
    sa = sorted(a)
    return sa[f] * (c - k) + sa[c] * (k - f)


print("frames:", len(bx))
print("bx range:", min(bx), "→", max(bx))
print("by range:", min(by), "→", max(by))
print(
    "spd px/frame: min/med/p95/max =",
    f"{min(spd):.2f}",
    f"{stats.median(spd):.2f}",
    f"{pct(spd, 95):.2f}",
    f"{max(spd):.2f}",
)

# flag spikes (likely decoy jumps)
th = max(12.0, pct(spd, 97))  # conservative
bad = [i + 1 for i, s in enumerate(spd) if s > th]
print("spike-threshold:", th, "spikes:", len(bad))
if bad[:12]:
    print("first spikes at frames:", bad[:12])
