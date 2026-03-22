"""Compare v22d vs v22e diagnostic CSVs at key game moments."""
import csv, os

BASE = r'D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC'

def load(name):
    rows = []
    with open(os.path.join(BASE, name)) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

d = load('002__v22d_tuned.diag.csv')
e = load('002__v22e_direct.diag.csv')

cols = list(d[0].keys())
print('Columns:', cols)
print(f'v22d: {len(d)} rows, v22e: {len(e)} rows')

# Identify the ball_x and cam_x column names
ball_col = 'ball_cx' if 'ball_cx' in cols else 'ball_x'
cam_col = 'cam_cx' if 'cam_cx' in cols else 'camera_cx'
src_col = 'source' if 'source' in cols else 'src'
print(f'Using: ball={ball_col}, cam={cam_col}, src={src_col}')
print()

events = [
    ("Keeper throw",        0,  30),
    ("Defender receives",   25, 50),
    ("Defender dribbles",   50, 90),
    ("Long ball kicked",   120,140),
    ("Long ball flight",   140,170),
    ("Receiver gets ball", 170,200),
    ("Build-up play",      200,260),
    ("Cross approach",     260,290),
    ("Cross delivered",    290,320),
    ("Shot taken",         320,340),
    ("Keeper save area",   340,360),
    ("Pan-hold zone",      332,425),
    ("Pan-hold to YOLO",   460,480),
]

print(f"{'Event':<22} {'Frames':<10} {'v22d gap':<10} {'v22e gap':<10} {'Change':<10}")
print("-" * 65)

for name, f_start, f_end in events:
    d_gaps = []
    e_gaps = []
    for i in range(f_start, min(f_end+1, len(d), len(e))):
        bx_d = float(d[i].get(ball_col, 0))
        cx_d = float(d[i].get(cam_col, 0))
        bx_e = float(e[i].get(ball_col, 0))
        cx_e = float(e[i].get(cam_col, 0))
        d_gaps.append(abs(bx_d - cx_d))
        e_gaps.append(abs(bx_e - cx_e))

    if d_gaps and e_gaps:
        d_avg = sum(d_gaps)/len(d_gaps)
        e_avg = sum(e_gaps)/len(e_gaps)
        delta = e_avg - d_avg
        pct = (delta/d_avg*100) if d_avg > 0 else 0
        sign = "+" if delta > 0 else ""
        print(f"{name:<22} f{f_start}-{f_end:<5} {d_avg:>7.0f}px  {e_avg:>7.0f}px  {sign}{delta:>.0f}px ({sign}{pct:.0f}%)")

print()
print("Camera cx at 1s intervals:")
print(f"{'Frame':<8} {'Ball x':<10} {'v22d cx':<10} {'v22e cx':<10} {'v22d gap':<10} {'v22e gap':<10}")
print("-" * 60)
for i in range(0, min(len(d), len(e)), 30):
    bx = float(e[i].get(ball_col, 0))
    cx_d = float(d[i].get(cam_col, 0))
    cx_e = float(e[i].get(cam_col, 0))
    gd = abs(bx - cx_d)
    ge = abs(bx - cx_e)
    print(f"f{i:<6} {bx:>7.0f}   {cx_d:>7.0f}   {cx_e:>7.0f}   {gd:>7.0f}px  {ge:>7.0f}px")

# Camera speed at key transitions
print()
print("Camera speed (px/frame) at transitions:")
transitions = [(20,35,"Throw receive"), (120,145,"Long ball kick"), (285,315,"Cross"), (315,340,"Shot")]
for f_s, f_e, nm in transitions:
    ds = []; es = []
    for i in range(f_s, min(f_e, len(d)-1, len(e)-1)):
        ds.append(abs(float(d[i+1][cam_col]) - float(d[i][cam_col])))
        es.append(abs(float(e[i+1][cam_col]) - float(e[i][cam_col])))
    if ds and es:
        print(f"  {nm:<20} v22d: avg={sum(ds)/len(ds):.1f} max={max(ds):.1f}  v22e: avg={sum(es)/len(es):.1f} max={max(es):.1f}")
