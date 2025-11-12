import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("telemetry", help="Path to JSONL telemetry")
args = ap.parse_args()

path = args.telemetry
xs, ys, ws, hs = [], [], [], []
first = last = None

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        x0,y0,w,h = d.get("crop", [None,None,None,None])
        if None in (x0,y0,w,h): continue
        if first is None: first = (i,x0,y0,w,h)
        last = (i,x0,y0,w,h)
        xs.append(x0); ys.append(y0); ws.append(w); hs.append(h)

if not xs:
    print("No crop in telemetry (check that render wrote 'crop' each frame)."); sys.exit(2)

print(f"frames with crop: {len(xs)}")
print(f"x0 min/max: {min(xs):.2f} / {max(xs):.2f}")
print(f"y0 min/max: {min(ys):.2f} / {max(ys):.2f}")
print(f"w  min/max: {min(ws):.2f} / {max(ws):.2f}")
print(f"h  min/max: {min(hs):.2f} / {max(hs):.2f}")
print("first frame crop:", first)
print(" last frame crop:", last)
