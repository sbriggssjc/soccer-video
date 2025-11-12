import sys, csv, math, numpy as np
if len(sys.argv)<3: raise SystemExit("usage: plan_kalman_fast.py <track_csv> <out_ps1>")
in_csv, out_ps1 = sys.argv[1], sys.argv[2]

# Load CSV (n,cx,cy,conf,w,h,fps)
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

# --- Constant-Velocity Kalman ---
def kalman_cv(cx, cy, conf, dt, q=6.0, r_base=9.0):
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

    # initialize at first measurement
    X[:2] = [cx[0], cy[0]]
    for i in range(len(cx)):
        # predict
        X = F @ X
        P = F @ P @ F.T + Q
        # update
        z = np.array([cx[i], cy[i]])
        confi = max(0.05, min(1.0, conf[i]))
        R = np.eye(2) * (r_base / confi)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        y = z - (H @ X)
        X = X + K @ y
        I = np.eye(4); P = (I - K @ H) @ P
        xs[i], ys[i], vx[i], vy[i] = X[0], X[1], X[2], X[3]
    return xs, ys, vx, vy

xs, ys, vx, vy = kalman_cv(CX, CY, CONF, dt)

speed = np.hypot(vx, vy)                   # px/s

# Predictive lead: higher base + more on sprints
base_lead_s = 0.40                          # was 0.32
extra_lead_s = np.clip((speed - 220.0)/350.0, 0, 1) * 0.30
lead_s = base_lead_s + extra_lead_s
lead_frames = np.clip((lead_s/dt).round().astype(int), 0, 48)

px = np.copy(xs); py = np.copy(ys)
for i in range(T):
    k = int(lead_frames[i])
    px[i] = xs[i] + vx[i]* (k*dt)
    py[i] = ys[i] + vy[i]* (k*dt)

px = np.clip(px, 1, w-2)
py = np.clip(py, 1, h-2)

# Smoothing: a bit less than before ? quicker reaction
def ema(a, alpha):
    o = np.copy(a)
    for i in range(1,len(a)): o[i]=alpha*a[i]+(1-alpha)*o[i-1]
    return o
px = ema(ema(px,0.44),0.28)
py = ema(ema(py,0.44),0.28)

# Hard rate caps: allow faster pan & accel (to not lose the ball)
def rate_cap(p, dv_max=10.0, da_max=0.65):   # was 6.0 / 0.35
    out = np.copy(p)
    for i in range(1,len(out)):
        d = out[i]-out[i-1]
        if d >  dv_max: out[i] = out[i-1] + dv_max
        if d < -dv_max: out[i] = out[i-1] - dv_max
    for i in range(2,len(out)):
        a = (out[i]-2*out[i-1]+out[i-2])
        if a >  da_max: out[i] = 2*out[i-1]-out[i-2] + da_max
        if a < -da_max: out[i] = 2*out[i-1]-out[i-2] - da_max
    return out
px = rate_cap(px)
py = rate_cap(py)

# Very short warm-up (prevents initial miss)
warm = min(8, T-1)                           # was 36
aw = np.linspace(0,1,warm)
px[:warm] = px[:warm]*aw + (w/2.0)*(1-aw)
py[:warm] = py[:warm]*aw + (h/2.0)*(1-aw)

# -------- Zoom: WIDEN when speed ? or confidence ? (opposite of before) --------
spd_s = ema(speed,0.30)
conf_s = ema(CONF,0.35)

# Compute tight zoom need from margins
mx_tight, my_tight = 160.0, 190.0           # less margin ? less forced zoom-in
dx = np.minimum(px, w-px) - mx_tight
dy = np.minimum(py, h-py) - my_tight
dx = np.clip(dx,1,None); dy = np.clip(dy,1,None)
safety = 1.08
z_need_x = (h*9/16)/(safety*2*dx)
z_need_y = (h)/(safety*2*dy)
z_need = np.maximum(1.0, np.maximum(z_need_x, z_need_y))

# Wide-on-speed/low-conf: blend toward 1.0 (wider FOV), not tighter
wide_bias = np.clip((spd_s-200.0)/320.0, 0, 1) + np.clip((0.35-conf_s)/0.35, 0, 1)*0.8
wide_bias = np.clip(wide_bias, 0, 1)        # 0..1
alpha = 0.70 - 0.55*wide_bias               # alpha? as speed?/conf? (more wide)
alpha = np.clip(alpha, 0.25, 0.70)

# Final zoom: mix (need vs. 1.0) + mild EMA + rate cap + lower ceiling
z = alpha*z_need + (1-alpha)*1.0
z = ema(z, 0.30)
z = np.clip(z, 1.0, 1.35)                   # was 1.45

def zoom_cap(z, dz_max=0.015):              # a hair looser than before
    out=np.copy(z)
    for i in range(1,len(out)):
        d = out[i]-out[i-1]
        if d >  dz_max: out[i] = out[i-1] + dz_max
        if d < -dz_max: out[i] = out[i-1] - dz_max
    return out
z = zoom_cap(z)

# Fit normalized polynomials (stable)
t = (N.astype(float)) / max(1, (T-1))
def poly_expr_from_fit(t, y, deg):
    P = np.polyfit(t, y, deg)
    d = len(P)-1
    terms=[]
    for i,a in enumerate(P):
        k = d-i
        coef = f"{a:.10g}"
        if k==0: terms.append(f"({coef})")
        elif k==1: terms.append(f"({coef})*(n/{T-1:.10g})")
        else:      terms.append(f"({coef})*pow(n/{T-1:.10g},{k})")
    return "(" + "+".join(terms) + ")"

# Lower degrees (reduce wiggle / overfit)
cx_expr = poly_expr_from_fit(t, px, deg=4)  # was 5
cy_expr = poly_expr_from_fit(t, py, deg=4)  # was 5
z_expr  = "clip(" + poly_expr_from_fit(t, z,  deg=3) + ",1.0,1.35)"

with open(out_ps1,"w",encoding="utf-8") as f:
    f.write(f"$cxExpr = '={cx_expr}'\n")
    f.write(f"$cyExpr = '={cy_expr}'\n")
    f.write(f"$zExpr  = '={z_expr}'\n")
