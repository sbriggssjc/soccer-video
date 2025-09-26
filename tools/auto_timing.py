import sys, json, pandas as pd, numpy as np

csv = sys.argv[1]
df = pd.read_csv(csv)
for c in ["t","ball_x","ball_y"]:
    if c not in df: raise SystemExit(json.dumps({"err": f"missing {c}"}))

# ball speed (px/s)
vx = np.gradient(df["ball_x"].values, df["t"].values, edge_order=2)
vy = np.gradient(df["ball_y"].values, df["t"].values, edge_order=2)
speed = np.hypot(vx, vy)

t0, t1 = float(df["t"].min()), float(df["t"].max())
cut     = t0 + 0.40*(t1 - t0)                # ignore early noise
mask    = df["t"].values >= cut
i_peak  = int(np.nanargmax(np.where(mask, speed, -1)))
t_shot  = float(df.loc[i_peak, "t"])

# knobs (pre-roll longer to capture throw/pass/dribble)
pre_roll   = 6.2     # try 5.5?7.0 to taste
postB      = 1.2     # B ends shortly after strike
postC_tail = 4.5     # keep following for the celebration

tA_end = max(0.0, t_shot - pre_roll)   # end of A
tB_end = min(t1 - 0.05, t_shot + postB)
tC_end = min(t1, tB_end + postC_tail)

def med_x(lo, hi, fallback=960.0):
    s = df[(df["t"]>=lo) & (df["t"]<=hi)]
    return float(np.median(s["ball_x"])) if len(s) else fallback

midxA  = med_x(max(t0, tA_end-2.5), tA_end)          # where throw/pass starts
midxB  = med_x(max(t0, t_shot-0.5), t_shot+0.8)      # shooter / goal channel
midxC  = med_x(t_shot+0.6, min(t1, t_shot+3.0), midxB)  # early celebration
midxD  = med_x(min(t1, tC_end), min(t1, tC_end+2.0), midxC)  # late celebration

clip = lambda x: float(np.clip(x, 320, 1600))
out = dict(
    t1 = round(tA_end,3),
    t2 = round(tB_end,3),
    t3 = round(tC_end,3),
    midxA = round(clip(midxA),1),
    midxB = round(clip(midxB),1),
    midxC = round(clip(midxC),1),
    midxD = round(clip(midxD),1),
)
print(json.dumps(out))
