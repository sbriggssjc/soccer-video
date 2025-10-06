import sys, csv, numpy as np

if len(sys.argv)<3: raise SystemExit("usage: auto_tune_plan_iter.py <track_csv> <out_vars>")
in_csv, out_ps1 = sys.argv[1], sys.argv[2]

N=[]; cx=[]; cy=[]; conf=[]; vx=[]; vy=[]; W=H=FPS=None
with open(in_csv,"r",newline="") as f:
    rd=csv.DictReader(f)
    cols=rd.fieldnames
    has_v = ("vx" in cols and "vy" in cols)
    for r in rd:
        N.append(int(r["n"]))
        cx.append(float(r["cx"])); cy.append(float(r["cy"])); conf.append(float(r["conf"]))
        if has_v:
            vx.append(float(r["vx"])); vy.append(float(r["vy"]))
        W=int(r.get("w",W or 0)); H=int(r.get("h",H or 0)); FPS=float(r.get("fps",FPS or 60.0))
N=np.array(N); cx=np.array(cx); cy=np.array(cy); conf=np.array(conf)
if len(N)==0: raise SystemExit("Empty CSV")

dt=1.0/max(FPS,1.0)
if len(vx)==len(N):
    vx=np.array(vx); vy=np.array(vy)
else:
    vx=np.gradient(cx, dt); vy=np.gradient(cy, dt)

# exponential moving average to tame noise (light)
def ema(a,a1):
    o=a.copy()
    for i in range(1,len(a)): o[i]=a1*a[i]+(1-a1)*o[i-1]
    return o

cx_s, cy_s = ema(cx,0.25), ema(cy,0.25)
vx_s, vy_s = ema(vx,0.25), ema(vy,0.25)

# confidence-adaptive look-ahead (short if low conf; long if high)
conf_s = ema(conf, 0.2)
LA_lo, LA_hi = 0.10, 0.50     # seconds
LA_s = LA_lo + (LA_hi-LA_lo)*np.clip((conf_s-0.15)/0.6, 0, 1)
lead_x = cx_s + vx_s*LA_s
lead_y = cy_s + vy_s*LA_s

# gap-aware widening: if conf<0.25, widen margins; also clamp within frame
spd = np.hypot(vx_s, vy_s)
vnorm = np.clip(spd/1200.0, 0, 1)
base_mx, base_my = 160.0, 200.0
mx = base_mx + 180.0*vnorm + 200.0*(conf_s<0.25)
my = base_my + 220.0*vnorm + 220.0*(conf_s<0.25)

cx_t = np.clip(lead_x, 0, W-1)
cy_t = np.clip(lead_y, 0, H-1)

# needed zoom from margins
dx = np.minimum(cx_t, W-cx_t) - mx
dy = np.minimum(cy_t, H-cy_t) - my
dx = np.clip(dx, 1, None); dy = np.clip(dy, 1, None)
safety = 1.08
z_need_x = (H*9/16)/(safety*2*dx)
z_need_y = (H)/(safety*2*dy)
z_needed = np.maximum(1.0, np.maximum(z_need_x, z_need_y))

# smoother zoom, with panic if needed
z_plan = np.minimum(np.maximum(1.0, 0.85*z_needed + 0.20), 1.95)
z_plan = ema(z_plan, 0.08)
excess = np.maximum(z_needed - z_plan, 0.0)
alpha  = np.clip(excess/0.06, 0, 1)
z_final= np.minimum(z_plan*(1-alpha) + z_needed*alpha, 2.05)

# piecewise compact expressions for ffmpeg eval (poly per 20 frames)
def piecewise(n,y,seg=20,deg=3):
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

cx_if = seg_if(piecewise(N,cx_t,seg=20,deg=3), "in_w/2")
cy_if = seg_if(piecewise(N,cy_t,seg=20,deg=3), "in_h/2")
z_if  = seg_if(piecewise(N,z_final,seg=20,deg=2), "1")

with open(out_ps1,"w",encoding="utf-8") as f:
    f.write(f"$cxExpr = '={cx_if}'\n")
    f.write(f"$cyExpr = '={cy_if}'\n")
    f.write(f"$zExpr  = '=clip({z_if},1.0,2.10)'\n")
    f.write(f"$Safety = 1.08\n")
