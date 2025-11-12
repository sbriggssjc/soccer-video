import sys, csv, numpy as np
try:
    from numpy import RankWarning
except Exception:
    class RankWarning(UserWarning): pass
import warnings
warnings.filterwarnings("ignore", category=RankWarning)

in_csv, out_ps1 = sys.argv[1], sys.argv[2]
N=[]; cx=[]; cy=[]; conf=[]; W=H=FPS=None
with open(in_csv,"r",newline="") as f:
    rd=csv.DictReader(f)
    for r in rd:
        N.append(int(r["n"]))
        cx.append(float(r["cx"])); cy.append(float(r["cy"])); conf.append(float(r["conf"]))
        W=int(r["w"]); H=int(r["h"]); FPS=float(r["fps"])
N=np.array(N); cx=np.array(cx); cy=np.array(cy); conf=np.array(conf)
if len(N)==0: raise SystemExit("Empty CSV")
dt = 1.0/max(FPS,1.0)

def ema(a,alpha):
    o=a.copy()
    for i in range(1,len(a)): o[i]=alpha*a[i]+(1-alpha)*o[i-1]
    return o

# lighter smoothing to reduce phase lag
cx_s, cy_s = ema(cx,0.26), ema(cy,0.26)
cx_s, cy_s = ema(cx_s,0.15), ema(cy_s,0.15)

# velocities & accelerations
vx = np.gradient(cx_s, dt); vy = np.gradient(cy_s, dt)
ax = np.gradient(vx, dt);   ay = np.gradient(vy, dt)

# ---- LEAD PREDICTION (increase if still behind) ----
LA_s     = 2.20    # seconds of lead (?)
ACC_GAIN = 0.25    # accel influence (? slightly to avoid overshoot)
A_CLAMP  = 2600.0  # px/s^2
ax = np.clip(ax, -A_CLAMP, A_CLAMP); ay = np.clip(ay, -A_CLAMP, A_CLAMP)
cx_pred = cx_s + vx*LA_s + 0.5*ACC_GAIN*ax*(LA_s**2)
cy_pred = cy_s + vy*LA_s + 0.5*ACC_GAIN*ay*(LA_s**2)

# warm start
warm = int(round(0.05*FPS))
cx_t = np.where(N<warm, cx_s, cx_pred)
cy_t = np.where(N<warm, cy_s, cy_pred)

# ---- DEAD-ZONE & SLEW (increase GAIN / SLEW if trailing fast) ----
DZ_X, DZ_Y = 85.0, 100.0
GAIN = 0.65
SLEW = 60.0   # px per frame cap

def slew_follow(target, current, dz, gain, slew):
    out = current.copy()
    for i in range(1,len(current)):
        err = target[i]-out[i-1]
        step = gain*err*(1.8 if abs(err)>dz else 1.0)
        if step >  slew: step =  slew
        if step < -slew: step = -slew
        out[i] = out[i-1] + step
    return out

cx_ff = slew_follow(cx_t, cx_s, DZ_X, GAIN, SLEW)
cy_ff = slew_follow(cy_t, cy_s, DZ_Y, GAIN, SLEW)

# ---- MARGINS AND SAFETY (smaller = tighter crop/faster zoom) ----
spd = np.hypot(vx,vy); vnorm = np.clip(spd/1000.0, 0, 1)
base_mx, base_my = 160.0, 190.0
mx = base_mx + 120.0*vnorm + 110.0*(conf<0.25)
my = base_my + 140.0*vnorm + 110.0*(conf<0.25)

dx = np.minimum(cx_ff, W-cx_ff) - mx
dy = np.minimum(cy_ff, H-cy_ff) - my
dx = np.clip(dx, 1, None); dy = np.clip(dy, 1, None)
safety=1.08
z_need_x = (H*9/16)/(safety*2*dx)
z_need_y = (H)/(safety*2*dy)
z_needed = np.maximum(1.0, np.maximum(z_need_x, z_need_y))

# faster zoom catch-up
z_plan = np.minimum(np.maximum(1.0, 0.78*z_needed + 0.36), 2.50)
z_plan = ema(z_plan, 0.10)
excess = np.maximum(z_needed - z_plan, 0.0)
alpha  = np.clip(excess/0.038, 0, 1)
z_soft = z_plan*(1-alpha) + z_needed*alpha
z_hard = np.maximum(z_soft, np.minimum(2.60, z_needed*1.17))
z_final= np.minimum(z_hard, 2.60)

# ---- Piecewise fit ? ffmpeg expr ----
def piecewise(n,y,seg=12,deg=3):
    S=[]; i=0
    while i<len(n):
        j=min(i+seg-1, len(n)-1)
        nn=n[i:j+1].astype(float); yy=y[i:j+1].astype(float)
        d=min(deg, len(nn)-1) if len(nn)>1 else 1
        co=np.polyfit(nn, yy, d)
        S.append((int(n[i]), int(n[j]), co)); i=j+1
    return S

def Pexpr(co):
    d=len(co)-1; t=[]
    for k,a in enumerate(co):
        p=d-k; c=f"{a:.10g}"
        t.append(f"({c})" if p==0 else (f"({c})*n" if p==1 else f"({c})*n^{p}"))
    return "(" + "+".join(t) + ")"

def seg_if(segs, fallback):
    return "".join([f"if(between(n,{a},{b}),{Pexpr(co)}," for a,b,co in segs]) + fallback + (")"*len(segs))

cx_if = seg_if(piecewise(N,cx_ff,seg=12,deg=3), "in_w/2")
cy_if = seg_if(piecewise(N,cy_ff,seg=12,deg=3), "in_h/2")
z_if  = seg_if(piecewise(N,z_final,seg=12,deg=2), "1")

with open(out_ps1,"w",encoding="utf-8") as f:
    f.write(f"$cxExpr = '={cx_if}'\n")
    f.write(f"$cyExpr = '=={cy_if}'.replace('==','=')\n")
    f.write(f"$zExpr  = '=clip({z_if},1.0,2.60)'\n")
