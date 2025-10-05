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
    for i in range(1,len(a)):
        o[i]=alpha*a[i]+(1-alpha)*o[i-1]
    return o

# base smoothing (lighter to reduce lag)
cx_s, cy_s = ema(cx,0.28), ema(cy,0.28)
cx_s, cy_s = ema(cx_s,0.16), ema(cy_s,0.16)

# velocities & accel
vx = np.gradient(cx_s, dt); vy = np.gradient(cy_s, dt)
ax = np.gradient(vx, dt);   ay = np.gradient(vy, dt)

# predictive lead (larger) + small accel term with clamp
LA_s = 1.80         # lead seconds (↑ from 1.30)
ACC_GAIN = 0.35     # modest accel influence
AX_MAX = 2400.0     # px/s^2 clamp
ay = np.clip(ay, -AX_MAX, AX_MAX); ax = np.clip(ax, -AX_MAX, AX_MAX)

cx_pred = cx_s + vx*LA_s + 0.5*ACC_GAIN*ax*(LA_s**2)
cy_pred = cy_s + vy*LA_s + 0.5*ACC_GAIN*ay*(LA_s**2)

# warm start (nearly none)
warm = int(round(0.06*FPS))
cx_t = np.where(N<warm, cx_s, cx_pred)
cy_t = np.where(N<warm, cy_s, cy_pred)

# dead-zone + slew limit: pull center faster when ball exits a small inner box
DZ_X, DZ_Y = 90.0, 110.0   # inner box half sizes
GAIN = 0.55                 # proportional slew gain per frame
SLEW = 42.0                 # max pixels per frame move (cap)

def slew_follow(target, current, dz, gain, slew):
    out = current.copy()
    for i in range(1,len(current)):
        err = target[i]-out[i-1]
        if abs(err) <= dz:
            step = gain*err
        else:
            step = gain*err*1.8
        step = np.clip(step, -slew, slew)
        out[i] = out[i-1] + step
    return out

cx_ff = slew_follow(cx_t, cx_s, DZ_X, GAIN, SLEW)
cy_ff = slew_follow(cy_t, cy_s, DZ_Y, GAIN, SLEW)

# speed-adaptive margins (toned down so we don't stay too wide at speed)
spd = np.hypot(vx,vy); vnorm = np.clip(spd/1000.0, 0, 1)
base_mx, base_my = 180.0, 210.0
mx = base_mx + 140.0*vnorm + 120.0*(conf<0.25)
my = base_my + 160.0*vnorm + 130.0*(conf<0.25)

# needed zoom from margins (tighter safety)
dx = np.minimum(cx_ff, W-cx_ff) - mx
dy = np.minimum(cy_ff, H-cy_ff) - my
dx = np.clip(dx, 1, None); dy = np.clip(dy, 1, None)
safety=1.10
z_need_x = (H*9/16)/(safety*2*dx)
z_need_y = (H)/(safety*2*dy)
z_needed = np.maximum(1.0, np.maximum(z_need_x, z_need_y))

# zoom plan with faster catch-up
z_plan = np.minimum(np.maximum(1.0, 0.80*z_needed + 0.35), 2.10)
z_plan = ema(z_plan, 0.10)
excess = np.maximum(z_needed - z_plan, 0.0)
alpha  = np.clip(excess/0.045, 0, 1)   # quicker ramp
z_soft = z_plan*(1-alpha) + z_needed*alpha
z_hard = np.maximum(z_soft, np.minimum(2.40, z_needed*1.15))
z_final= np.minimum(z_hard, 2.40)

# Piecewise fit (denser segments for responsiveness)
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
    f.write(f"$zExpr  = '=clip({z_if},1.0,2.40)'\n")
