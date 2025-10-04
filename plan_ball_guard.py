import sys, csv, math, numpy as np
if len(sys.argv) < 3: raise SystemExit("usage: plan_ball_guard.py <track_csv> <out_ps1>")
in_csv, out_ps1 = sys.argv[1], sys.argv[2]

# Load track: n,cx,cy,conf,w,h,fps
N=[]; CX=[]; CY=[]; CONF=[]
w=h=fps=None
with open(in_csv,"r",newline="") as f:
    rd=csv.DictReader(f)
    for r in rd:
        N.append(int(r["n"]))
        CX.append(float(r["cx"])); CY.append(float(r["cy"])); CONF.append(float(r["conf"]))
        w = int(r["w"]); h = int(r["h"]); fps = float(r["fps"])
N=np.array(N); CX=np.array(CX); CY=np.array(CY); CONF=np.array(CONF)
T=len(N)
if T < 8: raise SystemExit("track too short")
dt = 1.0/max(1.0,(fps if fps>1e-6 else 24.0))

# --- Kalman constant-velocity ---
def kalman_cv(cx, cy, conf, dt, q=5.5, r_base=7.0):
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
    H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)

    xs=np.zeros(T); ys=np.zeros(T); vx=np.zeros(T); vy=np.zeros(T)
    X[:2] = [cx[0], cy[0]]
    for i in range(T):
        # predict
        X = F @ X
        P = F @ P @ F.T + Q
        # update
        z = np.array([cx[i], cy[i]])
        confi = max(0.05, min(1.0, conf[i]))
        R = np.eye(2) * (r_base / confi)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        y  = z - (H @ X)
        X  = X + K @ y
        I = np.eye(4); P = (I - K @ H) @ P
        xs[i], ys[i], vx[i], vy[i] = X[0], X[1], X[2], X[3]
    return xs, ys, vx, vy

xs, ys, vx, vy = kalman_cv(CX, CY, CONF, dt)
speed = np.hypot(vx, vy)

# --- Predictive lead (stronger) ---
base_lead_s   = 0.44
extra_lead_s  = np.clip((speed - 200.0)/280.0, 0, 1) * 0.32
lead_s        = base_lead_s + extra_lead_s
lead_frames   = np.clip(np.round(lead_s/dt).astype(int), 0, 56)

px = np.copy(xs); py = np.copy(ys)
for i in range(T):
    k = int(lead_frames[i])
    px[i] = xs[i] + vx[i]*(k*dt)
    py[i] = ys[i] + vy[i]*(k*dt)

# --- Directional bias toward play (include receiver/defender)
vnorm = np.maximum(speed, 1e-6)
ux = vx / vnorm; uy = vy / vnorm
ahead = np.clip(0.12*speed, 0, 72.0)   # pixels
px += ux * ahead
py += uy * ahead

px = np.clip(px, 1, w-2); py = np.clip(py, 1, h-2)

# --- Smoothing
def ema(a, a1): 
    o=np.copy(a)
    for i in range(1,len(a)): o[i]=a1*a[i]+(1-a1)*o[i-1]
    return o
px = ema(ema(px,0.42),0.26)
py = ema(ema(py,0.42),0.26)

# --- Burst catch-up: temporarily raise pan/accel caps when off-center
def rate_cap_with_burst(p, cx, cy, w, h, base_v=10.0, base_a=0.65, boost_v=6.0, boost_a=0.45):
    out = np.copy(p)
    prev_d = 0.0
    for i in range(1,len(out)):
        # center error normalized (max of x/y error against half-dims)
        ex = abs(cx[i]-out[i-1])/(w*0.5)
        ey = abs(cy[i]-out[i-1])/(h*0.5)
        e  = max(ex,ey)
        # burst boost (0..1)
        b  = np.clip((e-0.18)/0.22, 0, 1)    # kick in when off-center > ~18%
        dv = base_v + boost_v*b
        d  = out[i]-out[i-1]
        if d >  dv: out[i] = out[i-1] + dv
        if d < -dv: out[i] = out[i-1] - dv
        # accel cap with burst
        if i>=2:
            a = (out[i]-2*out[i-1]+out[i-2])
            da = base_a + boost_a*b
            if a >  da: out[i] = 2*out[i-1]-out[i-2] + da
            if a < -da: out[i] = 2*out[i-1]-out[i-2] - da
    return out

# Use ball track (CX,CY) as the “should be center” reference
px = rate_cap_with_burst(px, CX, CY, w, h)
py = rate_cap_with_burst(py, CY, CX, h, w)  # reuse with swapped dims

# --- Initial lock: tiny blend from exact ball to avoid visible pop
warm = min(6, T-1)
aw = np.linspace(0,1,warm)
px[:warm] = px[:warm]*aw + CX[:warm]*(1-aw)
py[:warm] = py[:warm]*aw + CY[:warm]*(1-aw)

# --- Zoom: error/uncertainty → widen (pull z toward 1.0)
# Tight need from margins (kept modest so we don't over-zoom)
mx, my = 150.0, 185.0
dx = np.minimum(px, w-px) - mx
dy = np.minimum(py, h-py) - my
dx = np.clip(dx,1,None); dy = np.clip(dy,1,None)
safety = 1.08
z_need = np.maximum(1.0, np.maximum((h*9/16)/(safety*2*dx), (h)/(safety*2*dy)))

# Error & confidence gates
e_x = np.abs(CX - px)/(w*0.5)
e_y = np.abs(CY - py)/(h*0.5)
e   = np.maximum(e_x, e_y)                   # 0..~1
conf = ema(CONF, 0.35)
speed_s = ema(speed, 0.32)

# Mix toward wide when off-center or low-conf
beta_err  = np.clip((e-0.12)/0.25, 0, 1)     # kick in earlier than before
beta_conf = np.clip((0.40 - conf)/0.40, 0, 1)
beta_spd  = np.clip((speed_s - 210.0)/300.0, 0, 1)
beta = np.clip(0.15 + 0.55*beta_err + 0.25*beta_conf + 0.25*beta_spd, 0, 0.85)

z = (1.0*beta) + ((1.0-beta)*z_need)         # toward 1.0 when beta↑
z = ema(z, 0.30)
z = np.clip(z, 1.0, 1.33)

# Gentle zoom slew
for i in range(1,T):
    dz = z[i]-z[i-1]
    if dz > 0.012: z[i] = z[i-1] + 0.012
    if dz < -0.016: z[i] = z[i-1] - 0.016    # faster widen than tighten

# --- Compact expressions (normalized time → stable)
t = N.astype(float)/max(1,(T-1))
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

cx_expr = poly_expr_from_fit(t, px, deg=4)
cy_expr = poly_expr_from_fit(t, py, deg=4)
z_expr  = "clip(" + poly_expr_from_fit(t, z,  deg=3) + ",1.0,1.33)"

with open(out_ps1,"w",encoding="utf-8") as f:
    f.write(f"$cxExpr = '={cx_expr}'\n")
    f.write(f"$cyExpr = '={cy_expr}'\n")
    f.write(f"$zExpr  = '={z_expr}'\n")
