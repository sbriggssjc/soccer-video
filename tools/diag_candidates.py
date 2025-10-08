import argparse, cv2, numpy as np, os, json, math


def eq(g):
    return cv2.equalizeHist(g)


def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def draw_circ(img, x, y, color, r=12, thick=2):
    cv2.circle(img, (int(x), int(y)), r, color, thick)


ap = argparse.ArgumentParser()
ap.add_argument("--in", required=True)
ap.add_argument("--planner", default="out\\render_logs\\tester_022__SHOT.ball.jsonl")
ap.add_argument("--telemetry", default="out\\render_logs\\tester_022__SHOT.jsonl")
ap.add_argument("--anchors", default="out\\render_logs\\tester_022__SHOT.anchors.jsonl")
ap.add_argument("--frames", default="0,60,120,240,360,480,600,720,840,930")
ap.add_argument("--outdir", default="out\\diag_cands")
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# load planner path if exists (to overlay bx/by)
path = {}
if os.path.exists(args.planner):
    for i, line in enumerate(open(args.planner, "r", encoding="utf-8")):
        try:
            d = json.loads(line)
            path[i] = (d.get("bx"), d.get("by"))
        except:
            pass

# load anchors
anchors = {}
if os.path.exists(args.anchors):
    for line in open(args.anchors, "r", encoding="utf-8"):
        d = json.loads(line)
        anchors[int(round(d["t"] * (30.0)))] = (d["bx"], d["by"])  # rough fps, ok for overlay

cap = cv2.VideoCapture(args.in)
assert cap.isOpened()
for s in [int(x.strip()) for x in args.frames.split(",") if x.strip()]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    ok, fr = cap.read()
    if not ok:
        continue
    canvas = fr.copy()
    # overlay planned bx/by
    if s in path and all(v is not None for v in path[s]):
        draw_circ(canvas, path[s][0], path[s][1], (0, 0, 255), r=10, thick=3)  # red dot planned ball
    # overlay anchor if nearby frame index
    if s in anchors:
        draw_circ(canvas, anchors[s][0], anchors[s][1], (0, 255, 255), r=12, thick=2)  # yellow = anchor

    # dump frame
    cv2.imwrite(os.path.join(args.outdir, f"{s:06d}_overlay.png"), canvas)

print("Wrote overlays to", args.outdir)
