import sys, json, pandas as pd, numpy as np

csv = sys.argv[1]
df = pd.read_csv(csv)
for c in ["t","ball_x","ball_y"]:
    if c not in df: raise SystemExit(json.dumps({"err": f"missing {c}"}))

t0, t1 = float(df["t"].min()), float(df["t"].max())

# --- knobs you can tweak ---
goal_left, goal_right = 840.0, 1080.0
goal_cx = (goal_left + goal_right) / 2.0
later_frac = 0.65        # only search in the last 35% of the clip
lane_margin = 120.0      # allow a bit wider than the posts
speed_quant = 0.85       # keep peaks above this quantile (robust to noise)
pre_roll   = 7.0         # longer to catch throw/pass/dribble
postB      = 1.6         # keep strike + first beat
postC_tail = 5.0         # follow celebration
# ---------------------------

# ball kinematics
vx = np.gradient(df["ball_x"].values, df["t"].values, edge_order=2)
vy = np.gradient(df["ball_y"].values, df["t"].values, edge_order=2)
speed = np.hypot(vx, vy)

# focus late segment
cut = t0 + later_frac*(t1 - t0)
m = df["t"].values >= cut

# keep samples inside a widened goal lane
lane_lo, lane_hi = goal_left - lane_margin, goal_right + lane_margin
lane = (df["ball_x"].values >= lane_lo) & (df["ball_x"].values <= lane_hi)

cand = m & lane & np.isfinite(speed)

if not np.any(cand):
    # fallback: late segment only
    cand = m & np.isfinite(speed)

# rank by (1) speed (2) closeness to goal center (tie-break)
dist = np.abs(df["ball_x"].values - goal_cx)
rank_speed = (speed - np.nanmin(speed[cand])) / (np.nanmax(speed[cand]) - np.nanmin(speed[cand]) + 1e-9)
rank_dist  = 1.0 - (dist - np.nanmin(dist[cand])) / (np.nanmax(dist[cand]) - np.nanmin(dist[cand]) + 1e-9)
score = 0.7*rank_speed + 0.3*rank_dist

# only consider "high" speeds to avoid soft passes
th = np.nanquantile(rank_speed[cand], speed_quant)
strong = cand & (rank_speed >= th)

if not np.any(strong):
    strong = cand

# pick the latest best candidate
idxs = np.where(strong)[0]
best = idxs[np.argmax(score[idxs])]
t_shot = float(df.loc[best, "t"])

# phase boundaries
A_end = max(0.0, t_shot - pre_roll)
B_end = min(t1 - 0.05, t_shot + postB)
C_end = min(t1, B_end + postC_tail)

# centers
def med_x(lo, hi, fallback=goal_cx):
    s = df[(df["t"]>=lo) & (df["t"]<=hi)]
    return float(np.median(s["ball_x"])) if len(s) else fallback

midxA = med_x(max(t0, A_end-2.5), A_end, goal_cx)
midxB = med_x(max(t0, t_shot-0.5), t_shot+0.8, goal_cx)
midxC = med_x(t_shot+0.6, min(t1, t_shot+3.0), midxB)
midxD = med_x(min(t1, C_end), min(t1, C_end+2.0), midxC)

clip = lambda x: float(np.clip(x, 320, 1600))
out = dict(
    t1 = round(A_end,3),
    t2 = round(B_end,3),
    t3 = round(C_end,3),
    midxA = round(clip(midxA),1),
    midxB = round(clip(midxB),1),
    midxC = round(clip(midxC),1),
    midxD = round(clip(midxD),1),
    t_shot = round(t_shot,3)
)
print(json.dumps(out))
