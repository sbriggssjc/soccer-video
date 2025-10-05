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

def ema(a,a1):
    o=a.copy()
    for i in range(1,len(a)): o[i]=a1*a[i]+(1-a1)*o[i-1]
    return o

cx2, cy2 = ema(ema(cx,0.35),0.18), ema(ema(cy,0.35),0.18)

dt = 1.0/max(FPS,1.0)
LA_s = 1.30
LA   = int(round(LA_s*FPS))
warm = int(round(0.15*FPS))
cx_la = np.concatenate([cx2[LA:], np.repeat(cx2[-1], LA)])
cy_la = np.concatenate([cy2[LA:], np.repeat(cy2[-1], LA)])
cx_t  = np.where(N<warm, cx2, cx_la)
cy_t  = np.where(N<warm, cy2, cy_la)

vx = np.gradient(cx_t, dt); vy = np.gradient(cy_t, dt)
spd = np.hypot(vx,vy)
vnorm = np.clip(spd/1000.0, 0, 1)
base_mx, base_my = 200.0, 240.0
mx = base_mx + 200.0*vnorm + 140.0*(conf<0.30)
my = base_my + 220.0*vnorm + 160.0*(conf<0.30)

dx = np.minimum(cx_t, W-cx_t) - mx
dy = np.minimum(cy_t, H-cy_t) - my
dx = np.clip(dx, 1, None); dy = np.clip(dy, 1, None)
safety=1.15
z_need_x = (H*9/16)/(safety*2*dx)
z_need_y = (H)/(safety*2*dy)
z_needed = np.maximum(1.0, np.maximum(z_need_x, z_need_y))

z_plan = np.minimum(np.maximum(1.0, 0.85*z_needed + 0.30), 1.95)
z_plan = ema(z_plan, 0.12)
excess = np.maximum(z_needed - z_plan, 0.0)
alpha  = np.clip(excess/0.06, 0, 1)
z_soft = z_plan*(1-alpha) + z_needed*alpha
z_hard = np.maximum(z_soft, np.minimum(2.20, z_needed*1.12))
z_final= np.minimum(z_hard, 2.20)

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
    f.write(f"$cyExpr = '=={cy_if}'.replace('==','=')\n")
    f.write(f"$zExpr  = '=clip({z_if},1.0,2.20)'\n")
