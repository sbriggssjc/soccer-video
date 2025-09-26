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

# --- windows (push A earlier to catch throw-in/pass/dribble) ---
pre_roll = 7.2    # longer to include throw + first/second pass
postB      = 1.0    # keep strike + beat
postC_tail = 4.5    # follow early celebration

tA_end = max(0.0, t_shot - pre_roll)
tB_end = min(t1 - 0.05, t_shot + postB)
tC_end = min(t1, tB_end + postC_tail)

# --- robust centers with winsorized median + continuity limits ---
def robust_center(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0: return np.nan
    q10, q90 = np.percentile(a, [10, 90])
    a = np.clip(a, q10, q90)
    return float(np.median(a))

def phase_center(df, lo, hi, fallback):
    s = df[(df["t"]>=lo) & (df["t"]<=hi)]
    m = robust_center(s["ball_x"]) if len(s) else np.nan
    if not np.isfinite(m): m = fallback
    return float(np.clip(m, 400, 1520))

def limit_jump(curr, prev, max_jump):
    return float(prev + np.clip(curr - prev, -max_jump, max_jump))

mA = phase_center(df, max(t0, tA_end-2.5), tA_end, fallback=goal_mid)

# Around-shot center; fallback to goal if data is sparse
seg = df[(df["t"]>=t_shot-0.6)&(df["t"]<=t_shot+0.9)]
valid_frac = np.isfinite(seg["ball_x"]).mean() if len(seg) else 0.0
mB_raw = phase_center(df, t_shot-0.6, t_shot+0.9, fallback=goal_mid)
mB = goal_mid if valid_frac < 0.40 else mB_raw
mB = limit_jump(mB, mA, 220)

mC_raw = phase_center(df, t_shot+0.4, min(t1, t_shot+2.2), fallback=mB)
mC = limit_jump(mC_raw, mB, 220)

mD_raw = phase_center(df, min(t1, tC_end), min(t1, tC_end+2.0), fallback=mC)
mD = limit_jump(mD_raw, mC, 300)

print(json.dumps(dict(
    t1=round(tA_end,3), t2=round(tB_end,3), t3=round(tC_end,3),
    midxA=round(mA,1), midxB=round(mB,1), midxC=round(mC,1), midxD=round(mD,1),
    zA=1.00, zB=1.06, zC=1.04, zD=1.00,
    t_shot=round(t_shot,3)
)))

