import sys, csv, numpy as np
if len(sys.argv)<3: raise SystemExit("usage: plan_widefirst.py track.csv out.ps1vars")
in_csv, out_ps1 = sys.argv[1], sys.argv[2]

# load
N=[]; X=[]; Y=[]; C=[]; w=h=fps=None
with open(in_csv,"r",newline="") as f:
    rd=csv.DictReader(f)
    for r in rd:
        N.append(int(r["n"])); X.append(float(r["cx"])); Y.append(float(r["cy"])); C.append(float(r["conf"]))
        w=int(r["w"]); h=int(r["h"]); fps=float(r["fps"])
N=np.array(N); X=np.array(X); Y=np.array(Y); C=np.array(C); T=len(N)
fps = fps if fps and fps>1e-6 else 24.0; dt=1.0/fps

# simple constant-velocity smooth + velocity
def ema(a,alpha):
    o=np.copy(a)
    for i in range(1,len(a)): o[i]=alpha*a[i]+(1-alpha)*o[i-1]
    return o
Xs=ema(X,0.32); Ys=ema(Y,0.32); Xs=ema(Xs,0.18); Ys=ema(Ys,0.18)
Vx=np.gradient(Xs, dt); Vy=np.gradient(Ys, dt)
spd = np.hypot(Vx,Vy)

# predictive lead (shorter in turns), but we’ll keep camera intentionally wide
turn = np.abs(np.gradient(np.unwrap(np.arctan2(Vy,Vx+1e-6)), dt))
lead_s = 0.28 + 0.30*np.clip((spd-140)/240,0,1) - 0.18*np.clip(turn/6,0,1)
lead_s = np.clip(lead_s, 0.18, 0.70)
tx = np.clip(Xs + Vx*lead_s, 1, w-2)
ty = np.clip(Ys + Vy*lead_s, 1, h-2)

# jerk-limited follow (faster on large error)
def follow(t, W, dt):
    cam=np.zeros_like(t); vel=np.zeros_like(t); cam[0]=t[0]
    for i in range(1,len(t)):
        e=t[i]-cam[i-1]; en=abs(e)/(0.5*W)
        vmax = 14 + 30*np.clip(en,0,1.5)
        amax = 0.8 + 1.6*np.clip(en,0,1.5)
        kp=0.18+0.22*np.clip(en,0,1.2); kv=0.12
        a = kp*e - kv*vel[i-1]
        a = np.clip(a, -amax, amax)
        vel[i] = np.clip(vel[i-1]+a, -vmax, vmax)
        cam[i] = cam[i-1]+vel[i]
    return cam
cx = follow(tx, w, dt); cy = follow(ty, h, dt)

# ---- Zoom plan: WIDE-FIRST ----
# margins big enough for ball+nearest player
mx,my = 200.0, 240.0
def z_needed(cx,cy):
    dx = np.minimum(cx, w-cx) - mx
    dy = np.minimum(cy, h-cy) - my
    dx = np.clip(dx, 1, None); dy = np.clip(dy, 1, None)
    safety=1.06
    zx=(h*9/16)/(safety*2*dx); zy=(h)/(safety*2*dy)
    return np.maximum(1.0, np.maximum(zx,zy))

# baseline wide bias + predictive widen with speed
z_base = np.full(T, 1.03)  # ~3% in; feels wide
z_need = z_needed(cx,cy)
z_pred = 1.00 + 0.12*np.clip((spd-140)/260,0,1)   # widen up to +0.12 with bursts
z = np.minimum(1.45, np.maximum(z_base, z_need, z_pred))

# emergency inclusion (keep ball ≥36px from edge)
edge=36.0
for i in range(T):
    cw = min((h*9/16)/z[i], w); ch = min(h/z[i], h)
    cx[i] = np.clip(cx[i], cw/2, w-cw/2)
    cy[i] = np.clip(cy[i], ch/2, h-ch/2)
    left=cx[i]-cw/2; right=cx[i]+cw/2; top=cy[i]-ch/2; bot=cy[i]+ch/2
    bx,by = X[i],Y[i]
    need=False
    if bx-left<edge:   cx[i]=min(w-cw/2, bx-(cw/2-edge)); need=True
    if right-bx<edge:  cx[i]=max(cw/2,   bx+(cw/2-edge)); need=True
    if by-top<edge:    cy[i]=min(h-ch/2, by-(ch/2-edge)); need=True
    if bot-by<edge:    cy[i]=max(ch/2,   by+(ch/2-edge)); need=True
    if need:
        # widen slightly if still tight
        z[i]=min(1.45, max(z[i], z_needed(cx[i],cy[i]) * 1.02))

# constrain zoom tighten rate (slower tighten, faster widen)
for i in range(1,T):
    dz = z[i]-z[i-1]
    if dz >  0.010: z[i]=z[i-1]+0.010  # tighten slowly
    if dz < -0.060: z[i]=z[i-1]-0.060  # widen quickly

# final smooth for centers
def ema(a,alpha):
    o=np.copy(a)
    for i in range(1,len(a)): o[i]=alpha*a[i]+(1-alpha)*o[i-1]
    return o
cx=ema(cx,0.20); cy=ema(cy,0.20)

# piecewise fits (3 segments) for stable ffmpeg exprs
def fit_pw(arr, deg_c=5, deg_z=3):
    T=len(arr); b0=0; b1=int(0.33*(T-1)); b2=int(0.67*(T-1)); b3=T-1
    def seg(n0,n1,a,deg):
        n=np.arange(n0,n1+1); u=(n-n0)/max(1,(n1-n0))
        P=np.polyfit(u,a[n0:n1+1],deg); d=len(P)-1
        U=f"((n-{n0})/{max(1,(n1-n0))})"
        terms=[]
        for i,c in enumerate(P):
            k=d-i; coef=f"{c:.10g}"
            if k==0: terms.append(f"({coef})")
            elif k==1: terms.append(f"({coef})*{U}")
            else:      terms.append(f"({coef})*pow({U},{k})")
        return "(" + "+".join(terms) + ")", n0, n1
    def pw(exprs, default):
        out=default
        for e,a,b in reversed(exprs):
            out=f"if(between(n,{a},{b}),{e},{out})"
        return out
    e1,a1,b1=seg(b0,b1,arr,deg_c)
    e2,a2,b2=seg(b1,b2,arr,deg_c)
    e3,a3,b3=seg(b2,b3,arr,deg_c)
    return pw([(e1,a1,b1),(e2,a2,b2),(e3,a3,b3)], f"({arr[-1]:.6g})")
cx_expr = fit_pw(cx,5); cy_expr = fit_pw(cy,5)
z_expr  = "clip(" + fit_pw(z,deg_c=3) + ",1.00,1.45)"
with open(out_ps1,"w",encoding="utf-8") as f:
    f.write(f"$cxExpr='={cx_expr}'`n$cyExpr='={cy_expr}'`n$zExpr='={z_expr}'`n")
