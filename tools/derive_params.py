import sys, json, pandas as pd, numpy as np

csv = sys.argv[1]
df = pd.read_csv(csv)

for c in ["t","ball_x","ball_y"]:
    if c not in df: raise SystemExit(json.dumps({"err": f"missing {c}"}))

# --- tuneables (make these bigger/smaller to include more/less) ---
pre_roll    = 6.5   # seconds BEFORE shot -> extend to include throw-in/first pass/dribble
post_roll_B = 1.3   # seconds AFTER shot for Phase B (strike + first beat)
# ---------------------------------------------------------------

# ball speed (px/s) to find the shot moment robustly
vx = np.gradient(df["ball_x"].values, df["t"].values, edge_order=2)
vy = np.gradient(df["ball_y"].values, df["t"].values, edge_order=2)
speed = np.hypot(vx, vy)

t_start = float(df["t"].min())
t_end   = float(df["t"].max())
cut     = t_start + 0.40*(t_end - t_start)   # ignore early warmup
mask    = df["t"] >= cut

i_peak = int(np.nanargmax(np.where(mask, speed, -1)))
t_shot = float(df.loc[i_peak, "t"])

t1 = max(0.0, t_shot - pre_roll)
t2 = min(t_end-0.10, t_shot + post_roll_B)

def med_x(tlo, thi):
    s = df[(df["t"]>=tlo) & (df["t"]<=thi)]
    return float(np.median(s["ball_x"])) if len(s) else 960.0

midxA  = med_x(max(0.0, t1-2.0), t1)              # pre-shot bias (throw/wing)
midxBC = med_x(max(0.0, t_shot-0.4), t_shot+1.6)  # around shot (keep shooter/goal)

midxA  = float(np.clip(midxA, 320, 1600))
midxBC = float(np.clip(midxBC, 320, 1600))

print(json.dumps(dict(
  t1=round(t1,3),
  t2=round(t2,3),
  midxA=round(midxA,1),
  midxBC=round(midxBC,1)
)))
