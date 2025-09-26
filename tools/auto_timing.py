import sys, json, pandas as pd, numpy as np

csv = sys.argv[1]
df = pd.read_csv(csv)
for c in ["t", "ball_x", "ball_y"]:
    if c not in df:
        raise SystemExit(json.dumps({"err": f"missing {c}"}))

# --- shot pick, biased to goal channel ---
goal_left, goal_right = 840, 1080
goal_mid = 0.5*(goal_left + goal_right)

t0, t1 = float(df["t"].min()), float(df["t"].max())
late_cut = t0 + 0.40*(t1 - t0)

near_goal = df["ball_x"].between(goal_left-80, goal_right+80)
late      = df["t"] >= late_cut

vx = np.gradient(df["ball_x"].values, df["t"].values, edge_order=2)
vy = np.gradient(df["ball_y"].values, df["t"].values, edge_order=2)
speed = np.hypot(vx, vy)

cand = np.where(late & near_goal, speed, -1)
i_peak = int(np.nanargmax(cand))
if cand[i_peak] < 0:
    i_peak = int(np.nanargmax(np.where(late, speed, -1)))

t_shot = float(df.loc[i_peak, "t"])

# --- knobs ---
pre_roll   = 8.6     # longer buildup (throw-in/second pass/dribble)
postB      = 1.8     # cover strike + first beat
postC_tail = 6.0     # cross + celebration follow

tA_end = max(t0, t_shot - pre_roll)
tB_end = min(t1 - 0.05, t_shot + postB)
tC_end = min(t1, tB_end + postC_tail)

# --- cross detection (lateral pivot) ---
vx = np.gradient(df["ball_x"].values, df["t"].values, edge_order=2)
ax = np.gradient(vx, df["t"].values, edge_order=2)

win_lo = max(t0, t_shot - 1.0)
win_hi = min(t1, t_shot + 0.8)
w = (df["t"].values >= win_lo) & (df["t"].values <= win_hi)
if w.any():
    i_cross = int(np.nanargmax(np.abs(ax * w)))
    t_cross = float(df.loc[i_cross, "t"])
else:
    t_cross = t_shot

def med_x(lo, hi, fallback=960.0):
    s = df[(df["t"]>=lo) & (df["t"]<=hi)]
    return float(np.median(s["ball_x"])) if len(s) else fallback

midxA  = med_x(max(t0, tA_end-2.5), tA_end)
midxB  = med_x(max(t0, t_shot-0.5), t_shot+0.8)
midxC1 = med_x(tB_end, max(tB_end, t_cross), midxB)
midxC2 = med_x(min(t1, t_cross), min(t1, t_cross+2.0), midxB)
midxD  = med_x(min(t1, tC_end), min(t1, tC_end+2.0), midxC2)

clip = lambda x: float(np.clip(x, 320, 1600))
out = dict(
    t1=round(tA_end,3), t2=round(tB_end,3), t3=round(tC_end,3),
    t_cross=round(t_cross,3),
    midxA=round(clip(midxA),1), midxB=round(clip(midxB),1),
    midxC1=round(clip(midxC1),1), midxC2=round(clip(midxC2),1),
    midxD=round(clip(midxD),1),
    zA=1.00, zB=1.06, zC=1.04, zD=1.00,
    t_shot=round(t_shot,3)
)
print(json.dumps(out))

