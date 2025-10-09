import json
import math
import sys
import statistics as stats
from collections.abc import Mapping, Sequence
from pathlib import Path

p = Path(r"out\render_logs\tester_022__SHOT.ball.jsonl")
if not p.exists():
    print("no ball path file:", p)
    sys.exit(2)


def pick_coords(obj):
    if isinstance(obj, Mapping):
        key_shapes = [
            ("bx_stab", "by_stab"),
            ("bx_raw", "by_raw"),
            ("bx", "by"),
        ]
        for kx, ky in key_shapes:
            if kx in obj and ky in obj:
                x, y = obj[kx], obj[ky]
                if x is not None and y is not None:
                    return x, y
        for value in obj.values():
            x, y = pick_coords(value)
            if x is not None and y is not None:
                return x, y
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            x, y = pick_coords(item)
            if x is not None and y is not None:
                return x, y
    return None, None


xs, ys = [], []
for line in p.open("r", encoding="utf-8"):
    d = json.loads(line)
    x, y = pick_coords(d)
    if x is None or y is None:
        continue
    xs.append(float(x))
    ys.append(float(y))

n = len(xs)
if n < 3:
    print("frames: 0")
    print("NO VALID BALL COORDS FOUND")
    sys.exit(2)

spd = [math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) for i in range(1, n)]
if not spd:
    print("empty speed")
    sys.exit(2)

if len(spd) >= 100:
    p95 = stats.quantiles(spd, n=100)[94]
else:
    idx = max(int(len(spd) * 0.95) - 1, 0)
    p95 = sorted(spd)[idx]

thr = max(12.0, p95 * 3.0)
spikes = [i for i, v in enumerate(spd, start=1) if v > thr]

print(f"frames: {n}")
print(f"bx range: {min(xs)} \u2192 {max(xs)}")
print(f"by range: {min(ys)} \u2192 {max(ys)}")
print(
    "spd px/frame: min/med/p95/max = %.2f %.2f %.2f %.2f"
    % (min(spd), stats.median(spd), p95, max(spd))
)
print("spike-threshold:", thr, "spikes:", len(spikes))
if spikes:
    print("first spikes at frames:", spikes[:12])
