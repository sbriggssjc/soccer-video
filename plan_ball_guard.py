import sys, csv, math, numpy as np
if len(sys.argv) < 3: raise SystemExit("usage: plan_ball_guard.py <track_csv> <out_ps1>")
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
T=len(N);  dt = 1.0/max(1.0, (fps if fps>1e-6 else 24.0))
if T < 8: raise SystemExit("track too short")

# --- Kalman CV filter ---
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
spd = np.hypot(vx,vy)

# --- Predictive lead + directional bias ---
lead_s = 0.50 + np.clip((spd-180.0)/260.0,0,1)*0.30     # 0.50s base → up to +0.30s
kfrm   = np.clip(np.round(lead_s/dt).astype(int),0,60)
px = xs + vx*(kfrm*dt); py = ys + vy*(kfrm*dt)
vnorm = np.maximum(spd,1e-6)
px += (vx/vnorm)*np.clip(0.16*spd,0,84.0)
py += (vy/vnorm)*np.clip(0.16*spd,0,84.0)
px = np.clip(px,1,w-2); py = np.clip(py,1,h-2)

# --- Smooth (fast + fine) ---
def ema(a,a1):
    o=np.copy(a)
    for i in range(1,len(a)): o[i]=a1*a[i]+(1-a1)*o[i-1]
    return o
px = ema(ema(px,0.44),0.28)
py = ema(ema(py,0.44),0.28)

# --- Burst catch-up when off-center ---
def burst_cap(p, cx_ref, wdim, base_v=12.0, base_a=0.7, boost_v=8.0, boost_a=0.6):
    out=np.copy(p)
    for i in range(1,len(out)):
        e = abs(cx_ref[i]-out[i-1])/(wdim*0.5)
        b = np.clip((e-0.15)/0.20,0,1)
        dv = base_v + boost_v*b
        d  = out[i]-out[i-1]
        if d> dv: out[i]=out[i-1]+dv
        if d<-dv: out[i]=out[i-1]-dv
        if i>=2:
            a = (out[i]-2*out[i-1]+out[i-2])
            da = base_a + boost_a*b
            if a> da: out[i]=2*out[i-1]-out[i-2]+da
            if a<-da: out[i]=2*out[i-1]-out[i-2]-da
    return out

px = burst_cap(px, CX, w)
py = burst_cap(py, CY, h)

# --- Initial lock (tiny blend for first frames) ---
warm=min(6,T-1); aw=np.linspace(0,1,warm)
px[:warm] = px[:warm]*aw + CX[:warm]*(1-aw)
py[:warm] = py[:warm]*aw + CY[:warm]*(1-aw)

# --- Zoom: demand from margins + error/uncertainty → widen ---
mx,my = 170.0, 210.0
dx = np.minimum(px, w-px) - mx
dy = np.minimum(py, h-py) - my
dx = np.clip(dx,1,None); dy = np.clip(dy,1,None)
safety=1.08
z_need = np.maximum(1.0, np.maximum((h*9/16)/(safety*2*dx), (h)/(safety*2*dy)))

ex = np.abs(CX-px)/(w*0.5); ey = np.abs(CY-py)/(h*0.5); e=np.maximum(ex,ey)
conf = ema(CONF,0.35); spds=ema(spd,0.32)

beta = np.clip(0.20 + 0.60*np.clip((e-0.10)/0.22,0,1) + 0.30*np.clip((0.45-conf)/0.45,0,1) + 0.25*np.clip((spds-210)/320,0,1), 0, 0.95)
z = (1.0*beta) + ((1.0-beta)*z_need)      # always trends wide when unsure
z = ema(z,0.30)
z = np.clip(z,1.0,1.35)
for i in range(1,T):
    dz=z[i]-z[i-1]
    if dz> 0.008: z[i]=z[i-1]+0.008      # slow tighten
    if dz<-0.030: z[i]=z[i-1]-0.030      # fast widen

# --- Compact ffmpeg expressions (normalized n) ---
t = N.astype(float)/max(1,(T-1))
def fit_expr(t,y,deg):
    P=np.polyfit(t,y,deg); d=len(P)-1; terms=[]
    for i,a in enumerate(P):
        k=d-i; c=f"{a:.10g}"; u=f"(n/{T-1:.10g})"
        if k==0: terms.append(f"({c})")
        elif k==1: terms.append(f"({c})*{u}")
        else: terms.append(f"({c})*pow({u},{k})")
    return "(" + "+".join(terms) + ")"
cx_expr = fit_expr(t,px,4)
cy_expr = fit_expr(t,py,4)
z_expr  = "clip(" + fit_expr(t,z,3) + ",1.0,1.35)"

with open(out_ps1,"w",encoding="utf-8") as f:
    f.write(f"$cxExpr = '={cx_expr}'\n")
    f.write(f"$cyExpr = '={cy_expr}'\n")
    f.write(f"$zExpr  = '={z_expr}'\n")
