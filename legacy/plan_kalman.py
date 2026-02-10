import sys, csv, math, numpy as np
if len(sys.argv)<3: raise SystemExit("usage: plan_kalman.py <track_csv> <out_ps1>")
in_csv, out_ps1 = sys.argv[1], sys.argv[2]

# Load CSV (expects: n,cx,cy,conf,w,h,fps)
N=[]; CX=[]; CY=[]; CONF=[]
w=h=fps=None
with open(in_csv,"r",newline="") as f:
    rd=csv.DictReader(f)
    for r in rd:
        N.append(int(r["n"]))
        CX.append(float(r["cx"])); CY.append(float(r["cy"]))
        CONF.append(float(r["conf"]))
        w = int(r["w"]); h = int(r["h"]); fps = float(r["fps"])
N=np.array(N); CX=np.array(CX); CY=np.array(CY); CONF=np.array(CONF)
T=len(N)
if T<8: raise SystemExit("track too short")

dt = 1.0/(fps if fps>1e-6 else 24.0)

# --- Constant-Velocity Kalman in 2D (x,y,vx,vy) ---
def kalman_cv(cx, cy, conf, dt, q=4.0, r_base=9.0):
    X = np.zeros((4,))                      # [x,y,vx,vy]
    P = np.eye(4)*1e3
    F = np.array([[1,0,dt,0],
                  [0,1,0,dt],
                  [0,0,1, 0],
                  [0,0,0, 1]], dtype=float)
    Q = np.array([[dt**4/4,0,dt**3/2,0],
                  [0,dt**4/4,0,dt**3/2],
                  [dt**3/2,0,dt**2,0],
                  [0,dt**3/2,0,dt**2]], dtype=float) * q
    H = np.array([[1,0,0,0],
                  [0,1,0,0]], dtype=float)
    xs = np.zeros((len(cx),))
    ys = np.zeros((len(cx),))
    vx = np.zeros((len(cx),))
    vy = np.zeros((len(cx),))
    for i in range(len(cx)):
        # predict
        X = F @ X
        P = F @ P @ F.T + Q
        # measurement
        z = np.array([cx[i], cy[i]])
        confi = max(0.05, min(1.0, conf[i]))
        R = np.eye(2) * (r_base / confi)    # lower conf -> higher noise
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        y = z - (H @ X)
        X = X + K @ y
        I = np.eye(4); P = (I - K @ H) @ P
        xs[i], ys[i], vx[i], vy[i] = X[0], X[1], X[2], X[3]
    return xs, ys, vx, vy

xs, ys, vx, vy = kalman_cv(CX, CY, CONF, dt)

speed = np.hypot(vx, vy)                   # px/s
# Predict ahead so camera "meets" the ball
base_lead_s = 0.32
extra_lead_s = np.clip((speed - 260.0)/420.0, 0, 1) * 0.22   # add up to +0.22s when sprinting
lead_s = base_lead_s + extra_lead_s
lead_frames = np.clip((lead_s/dt).round().astype(int), 0, 36)  # cap lead

px = np.copy(xs); py = np.copy(ys)
for i in range(T):
    k = int(lead_frames[i])
    j = min(T-1, i+k)
    # predict with current velocity
    px[i] = xs[i] + vx[i]* (k*dt)
    py[i] = ys[i] + vy[i]* (k*dt)

# Clamp prediction inside field
px = np.clip(px, 1, w-2)
py = np.clip(py, 1, h-2)

# Two-stage EMA to remove high-freq wiggles
def ema(a, alpha):
    o = np.copy(a)
    for i in range(1,len(a)): o[i]=alpha*a[i]+(1-alpha)*o[i-1]
    return o
px = ema(ema(px,0.50),0.32)
py = ema(ema(py,0.50),0.32)

# Hard per-frame pan speed + acceleration caps (in px/frame)
def rate_cap(p, dv_max=6.0, da_max=0.35):
    out = np.copy(p)
    # velocity cap
    for i in range(1,len(out)):
        d = out[i]-out[i-1]
        if d >  dv_max: out[i] = out[i-1] + dv_max
        if d < -dv_max: out[i] = out[i-1] - dv_max
    # accel cap
    for i in range(2,len(out)):
        a = (out[i]-2*out[i-1]+out[i-2])
        if a >  da_max: out[i] = 2*out[i-1]-out[i-2] + da_max
        if a < -da_max: out[i] = 2*out[i-1]-out[i-2] - da_max
    return out
px = rate_cap(px, dv_max=6.0, da_max=0.35)
py = rate_cap(py, dv_max=6.0, da_max=0.35)

# Start warm-up: blend from center ? target over ~36 frames
warm = min(36, T-1)
aw = np.linspace(0,1,warm)
px[:warm] = px[:warm]*aw + (w/2.0)*(1-aw)
py[:warm] = py[:warm]*aw + (h/2.0)*(1-aw)

# --- Zoom from adaptive margins (wider on speed or low conf) ---
spd_s = ema(speed,0.35)
conf_s = ema(CONF,0.40)
mx_base, my_base = 200.0, 230.0
boost = np.clip((spd_s-240.0)/260.0, 0, 1) + np.clip((0.30-conf_s)/0.30, 0, 1)*0.9
mx = mx_base*(1.0+0.60*boost)
my = my_base*(1.0+0.60*boost)
dx = np.minimum(px, w-px) - mx
dy = np.minimum(py, h-py) - my
dx = np.clip(dx,1,None); dy = np.clip(dy,1,None)
safety = 1.10
z_need_x = (h*9/16)/(safety*2*dx)
z_need_y = (h)/(safety*2*dy)
z = np.maximum(1.0, np.maximum(z_need_x, z_need_y))
# bias a bit wide, cap lower, then smooth
z = np.minimum(1.45, 0.80*z + 0.22)
z = ema(z,0.34)
# zoom rate limit (per source frame)
def zoom_cap(z, dz_max=0.010):
    out=np.copy(z)
    for i in range(1,len(out)):
        d = out[i]-out[i-1]
        if d >  dz_max: out[i] = out[i-1] + dz_max
        if d < -dz_max: out[i] = out[i-1] - dz_max
    return out
z = zoom_cap(z, dz_max=0.010)

# --- Fit one normalized polynomial per axis (stable, no segment seams) ---
t = (N.astype(float)) / max(1, (T-1))
def poly_expr_from_fit(t, y, deg):
    P = np.polyfit(t, y, deg)        # highest power first
    d = len(P)-1
    terms=[]
    for i,a in enumerate(P):
        k = d-i
        coef = f"{a:.10g}"
        if k==0: terms.append(f"({coef})")
        elif k==1: terms.append(f"({coef})*(n/{T-1:.10g})")
        else:      terms.append(f"({coef})*pow(n/{T-1:.10g},{k})")
    return "(" + "+".join(terms) + ")"

cx_expr = poly_expr_from_fit(t, px, deg=5)
cy_expr = poly_expr_from_fit(t, py, deg=5)
z_expr  = "clip(" + poly_expr_from_fit(t, z,  deg=3) + ",1.0,1.45)"

with open(out_ps1,"w",encoding="utf-8") as f:
    f.write(f"$cxExpr = '={cx_expr}'\n")
    f.write(f"$cyExpr = '={cy_expr}'\n")
    f.write(f"$zExpr  = '={z_expr}'\n")
