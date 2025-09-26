import sys, json, pandas as pd, numpy as np

csv = sys.argv[1]
df = pd.read_csv(csv)
for c in ["t","ball_x","ball_y"]:
    if c not in df: raise SystemExit(json.dumps({"err": f"missing {c}"}))

t0, t1 = float(df["t"].min()), float(df["t"].max())

goal_left, goal_right = 840.0, 1080.0
goal_mid = 0.5 * (goal_left + goal_right)

pre_roll   = 6.2
postB      = 1.2
postC_tail = 4.5

# ball kinematics
vx = np.gradient(df["ball_x"].values, df["t"].values, edge_order=2)
vy = np.gradient(df["ball_y"].values, df["t"].values, edge_order=2)
speed = np.hypot(vx, vy)

# bias the shot detection toward the goal channel, but keep a robust fallback
late_cut = t0 + 0.40 * (t1 - t0)
near_goal = df["ball_x"].between(goal_left - 80, goal_right + 80)
late = df["t"] >= late_cut
finite_speed = np.isfinite(speed)

cand = np.where(late & near_goal & finite_speed, speed, -1)
i_peak = int(np.nanargmax(cand))
if cand[i_peak] < 0:
    i_peak = int(np.nanargmax(np.where(late & finite_speed, speed, -1)))

t_shot = float(df.loc[i_peak, "t"])

# phase boundaries
tA_end = max(0.0, t_shot - pre_roll)
tB_end = min(t1 - 0.05, t_shot + postB)
tC_end = min(t1, tB_end + postC_tail)


def robust_center(x, lo=400, hi=1520):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    q10, q90 = np.percentile(x, [10, 90])
    x = np.clip(x, q10, q90)
    return float(np.median(x))


def phase_center(df, t_lo, t_hi, fallback=960.0):
    s = df[(df["t"] >= t_lo) & (df["t"] <= t_hi)]
    m = robust_center(s["ball_x"]) if len(s) else np.nan
    if not np.isfinite(m):
        m = fallback
    return float(np.clip(m, 400, 1520))


def limit_jump(curr, prev, max_jump=220):
    return float(prev + np.clip(curr - prev, -max_jump, max_jump))


mA = phase_center(df, max(t0, tA_end - 2.5), tA_end, fallback=goal_mid)

mB_raw = phase_center(df, max(t0, t_shot - 0.6), t_shot + 0.9, fallback=goal_mid)
window = df.loc[(df["t"] >= t_shot - 0.6) & (df["t"] <= t_shot + 0.9), "ball_x"]
valid = np.isfinite(window)
valid_frac = float(valid.mean()) if len(window) else np.nan
if not np.isfinite(valid_frac) or valid_frac < 0.40:
    mB = goal_mid
else:
    mB = mB_raw
mB = limit_jump(mB, mA, 220)

mC_raw = phase_center(df, t_shot + 0.4, min(t1, t_shot + 2.2), fallback=mB)
mC = limit_jump(mC_raw, mB, 220)

mD_raw = phase_center(df, min(t1, tC_end), min(t1, tC_end + 2.0), fallback=mC)
mD = limit_jump(mD_raw, mC, 300)

clip_range = lambda x: float(np.clip(x, 400, 1520))

out = dict(
    t1=round(tA_end, 3),
    t2=round(tB_end, 3),
    t3=round(tC_end, 3),
    midxA=round(clip_range(mA), 1),
    midxB=round(clip_range(mB), 1),
    midxC=round(clip_range(mC), 1),
    midxD=round(clip_range(mD), 1),
    zA=1.00,
    zB=1.06,
    zC=1.04,
    zD=1.00,
    t_shot=round(t_shot, 3),
)
print(json.dumps(out))
