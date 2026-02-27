"""Compare v22d vs v22g diagnostic CSVs."""
import csv, os

BASE = r'D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC'

def load(name):
    rows = []
    with open(os.path.join(BASE, name)) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

d = load('002__v22d_tuned.diag.csv')
g = load('002__v22g_responsive.diag.csv')

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
    ("Keeper save",        340,360),
]

print(f"{'Event':<22} {'Frames':<10} {'v22d gap':<10} {'v22g gap':<10} {'Change':<12}")
print("-" * 65)

for name, f_start, f_end in events:
    d_gaps = []; g_gaps = []
    for i in range(f_start, min(f_end+1, len(d), len(g))):
        bx_d = float(d[i]['ball_x']); cx_d = float(d[i]['cam_cx'])
        bx_g = float(g[i]['ball_x']); cx_g = float(g[i]['cam_cx'])
        d_gaps.append(abs(bx_d - cx_d))
        g_gaps.append(abs(bx_g - cx_g))
    if d_gaps and g_gaps:
        da = sum(d_gaps)/len(d_gaps)
        ga = sum(g_gaps)/len(g_gaps)
        delta = ga - da
        sign = "+" if delta > 0 else ""
        pct = (delta/da*100) if da > 0 else 0
        print(f"{name:<22} f{f_start}-{f_end:<5} {da:>7.0f}px  {ga:>7.0f}px  {sign}{delta:>.0f}px ({sign}{pct:.0f}%)")

print()
print("Camera cx at 1s:")
print(f"{'Frame':<8} {'Ball':<8} {'v22d':<8} {'v22g':<8} {'d gap':<8} {'g gap':<8}")
print("-" * 50)
for i in range(0, min(len(d), len(g)), 30):
    bx = float(g[i]['ball_x'])
    cd = float(d[i]['cam_cx']); cg = float(g[i]['cam_cx'])
    print(f"f{i:<6} {bx:>6.0f}  {cd:>6.0f}  {cg:>6.0f}  {abs(bx-cd):>5.0f}px {abs(bx-cg):>5.0f}px")

# Camera speed at transitions
print()
print("Camera speed (px/f) at transitions:")
for f_s, f_e, nm in [(20,35,"Throw"), (125,145,"Long ball"), (285,315,"Cross"), (315,340,"Shot")]:
    ds=[]; gs=[]
    for i in range(f_s, min(f_e, len(d)-1, len(g)-1)):
        ds.append(abs(float(d[i+1]['cam_cx']) - float(d[i]['cam_cx'])))
        gs.append(abs(float(g[i+1]['cam_cx']) - float(g[i]['cam_cx'])))
    if ds and gs:
        print(f"  {nm:<16} v22d: avg={sum(ds)/len(ds):.1f} max={max(ds):.1f}  v22g: avg={sum(gs)/len(gs):.1f} max={max(gs):.1f}")
