import sys, csv, numpy as np
if len(sys.argv) < 3: raise SystemExit("usage: plan_adaptive_speed.py <track_csv> <out_ps1>")
in_csv, out_ps1 = sys.argv[1], sys.argv[2]

# Load track
N=[]; CX=[]; CY=[]; CONF=[]
w=h=fps=None
with open(in_csv,"r",newline="") as f:
    rd=csv.DictReader(f)
    for r in rd:
        N.append(int(r["n"]))
        CX.append(float(r["cx"])); CY.append(float(r["cy"])); CONF.append(float(r["conf"]))
        w = int(r["w"]); h = int(r["h"]); fps = float(r["fps"])
N=np.array(N); CX=np.array(CX); CY=np.array(CY); CONF=np.array(CONF)
T=len(N);  fps = fps if fps and fps>1e-6 else 24.0
dt = 1.0/fps
if T < 8: raise SystemExit("track too short")

# Exponential moving average helper
def ema(a,alpha):
    o=np.copy(a)
    for i in range(1,len(a)):
        o[i] = alpha*a[i] + (1-alpha)*o[i-1]
    return o

# --- Constant-velocity Kalman smooth ---
def kalman_cv(cx, cy, conf, dt, q=6.5, r_base=7.0):
    X = np.zeros((4,)); P = np.eye(4)*1e3
    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],float)
    Q = np.array([[dt**4/4,0,dt**3/2,0],[0,dt**4/4,0,dt**3/2],[dt**3/2,0,dt**2,0],[0,dt**3/2,0,dt**2]],float)*q
    H = np.array([[1,0,0,0],[0,1,0,0]],float)
    xs=np.zeros(T); ys=np.zeros(T); vx=np.zeros(T); vy=np.zeros(T)
    X[:2] = [cx[0], cy[0]]
    for i in range(T):
        X = F @ X; P = F @ P @ F.T + Q
        z = np.array([cx[i],cy[i]]); c = max(0.05,min(1.0, conf[i]))
        R = np.eye(2)*(r_base/c)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        X = X + K@(z - H@X); P = (np.eye(4)-K@H)@P
        xs[i],ys[i],vx[i],vy[i]=X
    return xs,ys,vx,vy

xs,ys,vx,vy = kalman_cv(CX,CY,CONF,dt)
ax = np.gradient(vx, dt)
ay = np.gradient(vy, dt)
spd = np.hypot(vx,vy)
vnorm = np.clip(spd/260.0, 0, 1)
conf_s = ema(CONF,0.35)

# --- Curvature (anticipate reversals) ---
# finite difference of heading
heading = np.arctan2(vy, vx + 1e-6)
dhead = np.diff(np.unwrap(heading))
dhead = np.concatenate([[0.0], dhead])
curv  = np.abs(dhead)/(dt+1e-9)            # rad/s

# --- Speed/curvature-based lookahead ---
lead_s = 0.35 + np.clip((spd-160.0)/240.0,0,1)*0.30 - np.clip(curv/6.0,0,1)*0.18
lead_s = np.clip(lead_s, 0.20, 0.75)
LA_s = lead_s
cx_s, cy_s = xs, ys
q = np.clip((conf_s - 0.20)/0.40, 0, 1)

# Motion-follow target for jittery detections
LA_motion = np.clip(0.45 + 0.35*vnorm, 0.3, 0.9)
cx_m = cx_s + vx*LA_motion
cy_m = cy_s + vy*LA_motion

# Blend detection lead vs motion-follow by confidence
cx_p = q*(cx_s + vx*LA_s + 0.5*ax*(LA_s**2)) + (1-q)*cx_m
cy_p = q*(cy_s + vy*LA_s + 0.5*ay*(LA_s**2)) + (1-q)*cy_m

# Predict target (ball) ahead
px_targ = np.clip(cx_p, 1, w-2)
py_targ = np.clip(cy_p, 1, h-2)

# --- Adaptive velocity/accel limits (gain schedule on error & speed) ---
def make_limits(err_norm, spd):
    # err_norm: error as fraction of half-span (0..1+), spd in px/s
    # Higher error and speed -> higher limits
    v_base = 10.0; v_err = 28.0; v_spd = 18.0
    a_base = 0.60; a_err = 1.40; a_spd = 0.90
    v_lim = v_base + v_err*np.clip(err_norm,0,1.4) + v_spd*np.clip((spd-160)/260,0,1)
    a_lim = a_base + a_err*np.clip(err_norm,0,1.4) + a_spd*np.clip((spd-160)/260,0,1)
    return v_lim, a_lim

# Jerk-limited S-curve controller (asymmetric accel/brake + reversal anticipation)
def s_curve_follow(target, width, dt):
    cam = np.zeros_like(target); vel = np.zeros_like(target)
    cam[0] = target[0]
    # preview sign changes (reversal anticipation)
    for i in range(1,len(cam)):
        e = target[i] - cam[i-1]
        err_norm = abs(e)/(0.5*width)
        v_lim, a_lim = make_limits(err_norm, spd[i])

        # anticipation: if target direction likely to flip soon, reduce allowed speed
        flip_factor = np.clip(curv[i]/5.0, 0.0, 1.0)   # 0 (straight) .. 1 (sharp turn)
        v_lim *= (1.0 - 0.45*flip_factor)

        # proportional assist (light PD)
        kp = 0.18 + 0.22*np.clip(err_norm,0,1.2)
        kv = 0.12
        a_cmd = kp*e - kv*vel[i-1]

        # asymmetric accel/brake
        if np.sign(a_cmd) == np.sign(e):
            a_cmd = np.clip(a_cmd, -a_lim, +a_lim*1.35)   # push harder when chasing
        else:
            a_cmd = np.clip(a_cmd, -a_lim*0.90, +a_lim)   # brake gentler to avoid snap

        # integrate with jerk limit (limit accel delta)
        if i>=2:
            jerk_lim = 0.60 + 0.90*np.clip(err_norm,0,1.4)
            da = np.clip(a_cmd - (vel[i-1]-vel[i-2]), -jerk_lim, +jerk_lim)
            a_cmd = (vel[i-1]-vel[i-2]) + da

        vel[i] = np.clip(vel[i-1] + a_cmd, -v_lim, +v_lim)
        cam[i] = cam[i-1] + vel[i]
    return cam

cx_cam = s_curve_follow(px_targ, w, dt)
cy_cam = s_curve_follow(py_targ, h, dt)

# Confidence-aware failsafe margins
mx_sym = np.full(T, 170.0, dtype=float)
my_sym = np.full(T, 210.0, dtype=float)
wide_boost = 1 + 0.45*(1-q)          # up to +45% wider when q is low
mx_sym *= wide_boost
my_sym *= wide_boost
mx_sym = np.clip(mx_sym, 1.0, 0.5*w-2.0)
my_sym = np.clip(my_sym, 1.0, 0.5*h-2.0)

for i in range(T):
    cx_cam[i] = np.clip(cx_cam[i], mx_sym[i], w - mx_sym[i])
    cy_cam[i] = np.clip(cy_cam[i], my_sym[i], h - my_sym[i])

# gentle finishing smooth (keeps path clean but responsive)
cx_cam = ema(cx_cam,0.22); cy_cam = ema(cy_cam,0.22)

# Re-apply failsafe after smoothing to honor widened margins
for i in range(T):
    cx_cam[i] = np.clip(cx_cam[i], mx_sym[i], w - mx_sym[i])
    cy_cam[i] = np.clip(cy_cam[i], my_sym[i], h - my_sym[i])

emergency_idx = None
if T > 0:
    fail_frames = int(round(6.0/dt)) if dt>0 else T
    sustain = max(1, int(round(0.8/dt))) if dt>0 else 1
    low_conf = conf_s < 0.12
    limit = max(0, T - sustain)
    for i in range(min(T, max(fail_frames,0)), limit):
        if np.all(low_conf[i:i+sustain]):
            emergency_idx = i
            break
    if emergency_idx is not None:
        center_x = w/2.0
        center_y = h/2.0
        blend = np.linspace(0.0, 1.0, T-emergency_idx, endpoint=True)
        cx_seg = cx_cam[emergency_idx:]
        cy_seg = cy_cam[emergency_idx:]
        cx_cam[emergency_idx:] = cx_seg*(1.0-blend) + center_x*blend
        cy_cam[emergency_idx:] = cy_seg*(1.0-blend) + center_y*blend
        for i in range(emergency_idx, T):
            cx_cam[i] = np.clip(cx_cam[i], mx_sym[i], w - mx_sym[i])
            cy_cam[i] = np.clip(cy_cam[i], my_sym[i], h - my_sym[i])

# --- Zoom plan: widen when fast, uncertain, or near edge ---
dx = np.minimum(cx_cam, w-cx_cam) - mx_sym
dy = np.minimum(cy_cam, h-cy_cam) - my_sym
dx = np.clip(dx,1,None); dy = np.clip(dy,1,None)
safety=1.08
z_need = np.maximum(1.0, np.maximum((h*9/16)/(safety*2*dx), (h)/(safety*2*dy)))

errx = np.abs(CX-cx_cam)/(0.5*w); erry = np.abs(CY-cy_cam)/(0.5*h)
errn = np.maximum(errx,erry)

wide_bias = 0.18 + 0.50*np.clip((errn-0.08)/0.20,0,1) \
                  + 0.28*np.clip((0.55-conf_s)/0.55,0,1) \
                  + 0.28*np.clip((np.hypot(vx,vy)-190)/300,0,1)
wide_bias = np.clip(wide_bias,0,0.95)

z = (1.0*wide_bias) + ((1.0-wide_bias)*z_need)
z = np.clip(ema(z,0.28),1.0,1.35)

if emergency_idx is not None:
    base_zoom = np.clip(1.10 + 0.35*(1 - q[emergency_idx:]), 1.0, 1.35)
    z[emergency_idx:] = np.maximum(z[emergency_idx:], base_zoom)

# limit tighten speed, allow quick widen
for i in range(1,T):
    dz=z[i]-z[i-1]
    if dz> 0.010: z[i]=z[i-1]+0.010
    if dz<-0.035: z[i]=z[i-1]-0.035

# --- Compact ffmpeg expressions over normalized time ---
t = N.astype(float)/max(1,(T-1))
def fit_expr(t,y,deg):
    P=np.polyfit(t,y,deg); d=len(P)-1; terms=[]; u=f"(n/{T-1:.10g})"
    for i,a in enumerate(P):
        k=d-i; c=f"{a:.10g}"
        if k==0: terms.append(f"({c})")
        elif k==1: terms.append(f"({c})*{u}")
        else: terms.append(f"({c})*pow({u},{k})")
    return "(" + "+".join(terms) + ")"
cx_expr = fit_expr(t,cx_cam,5)
cy_expr = fit_expr(t,cy_cam,5)
z_expr  = "clip(" + fit_expr(t,z,3) + ",1.0,1.35)"

with open(out_ps1,"w",encoding="utf-8") as f:
    f.write(f"$cxExpr = '={cx_expr}'\n")
    f.write(f"$cyExpr = '={cy_expr}'\n")
    f.write(f"$zExpr  = '={z_expr}'\n")
