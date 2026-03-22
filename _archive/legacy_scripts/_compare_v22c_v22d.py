import csv

def load(path):
    return list(csv.DictReader(open(path)))

c = load(r'D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__v22c_planner.diag.csv')
d = load(r'D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__v22d_tuned.diag.csv')

print("=== v22c vs v22d KEY EVENT COMPARISON ===")
events = [
    (0, "Keeper has ball"),
    (30, "Throw to defender"),
    (45, "Ball at defender"),
    (90, "Build-up play"),
    (125, "Pass forward begins"),
    (140, "Cross-field transition"),
    (150, "Ball arrives right"),
    (270, "Far right (boundary)"),
    (300, "Shot area"),
    (320, "Ball returning center"),
    (330, "Shot taken"),
    (360, "Post-shot hold"),
    (420, "Keeper save area"),
    (450, "Ball far left"),
    (466, "Ball reappears right"),
]
print(f"{'frame':>5} {'event':<25} {'ball_x':>6} {'c_cx':>6} {'c_gap':>6} {'d_cx':>6} {'d_gap':>6} {'improve':>8}")
for f, label in events:
    if f < len(c) and f < len(d):
        bx = float(d[f]['ball_x'])
        cc = float(c[f]['cam_cx'])
        dc = float(d[f]['cam_cx'])
        cg = abs(bx - cc)
        dg = abs(bx - dc)
        imp = f"{(cg-dg)/max(cg,1)*100:+.0f}%" if cg > 5 else "~same"
        print(f"{f:>5} {label:<25} {bx:>6.0f} {cc:>6.0f} {cg:>6.0f} {dc:>6.0f} {dg:>6.0f} {imp:>8}")

# Edge proximity comparison
print(f"\n=== FRAMES WITH BALL >50% TO EDGE ===")
print(f"  v22c: ", end="")
cnt_c = sum(1 for r in c if abs(float(r['ball_x'])-float(r['cam_cx'])) / max(float(r['crop_w'])/2, 1) > 0.50)
print(f"{cnt_c} frames")
print(f"  v22d: ", end="")
cnt_d = sum(1 for r in d if abs(float(r['ball_x'])-float(r['cam_cx'])) / max(float(r['crop_w'])/2, 1) > 0.50)
print(f"{cnt_d} frames")
print(f"  Improvement: {cnt_c - cnt_d} fewer edge-frames ({100*(cnt_c-cnt_d)/max(cnt_c,1):.0f}% reduction)")
