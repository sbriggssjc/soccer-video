import sys, csv, numpy as np
if len(sys.argv)<3: raise SystemExit("usage: plan_hyper.py track.csv out.ps1vars")
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

# knobs (raise SPEED if still slow; reduce ZMAX to keep wider)
SPEED = 2.6
ZMAX  = 1.35
EDGE  = 80.0      # keep-in distance near ball
MX,MY = 170.0,210.0  # margins for zoom compute

def ema(a,alpha):
    o=np.copy(a)
    for i in range(1,len(a)): o[i]=alpha*a[i]+(1-alpha)*o[i-1]
    return o

# light smoothing
Xs=ema(X,0.22); Ys=ema(Y,0.22)
Vx=np.gradient(Xs, dt); Vy=np.gradient(Ys, dt)
Ax=np.gradient(Vx, dt); Ay=np.gradient(Vy, dt)
spd = np.hypot(Vx,Vy)
acc = np.hypot(Ax,Ay)

# quadratic look-ahead: x + v*L + 0.5*a*L^2
baseL = 0.25 + 0.75*np.clip((spd-100)/260,0,1)   # up to ~1.0s
baseL *= (1.0 - 0.35*np.clip(acc/900,0,1))      # reduce lead on hard turns
L = np.clip(baseL*SPEED, 0.20, 1.20)

tx = np.clip(Xs + Vx*L + 0.5*Ax*(L**2), 1, w-2)
ty = np.clip(Ys + Vy*L + 0.5*Ay*(L**2), 1, h-2)

# jerk-limited follow + early SNAP
def follow(target, W, dt):
    cam=np.zeros_like(target); vel=np.zeros_like(target); cam[0]=target[0]
    for i in range(1,len(target)):
        e = target[i]-cam[i-1]
        en = abs(e)/(0.28*W)
        vmax = (42 + 120*np.clip(en,0,1.5))*SPEED     # px/frame
        amax = (2.2 + 5.5*np.clip(en,0,1.5))*SPEED    # px/frame^2
        kp=0.36+0.36*np.clip(en,0,1.5); kv=0.12
        if en>0.6:   # snap earlier
            vmax *= 1.5; amax *= 1.6; kp *= 1.2
        a = kp*e - kv*vel[i-1]
        a = np.clip(a, -amax, amax)
        vel[i] = np.clip(vel[i-1]+a, -vmax, vmax)
        cam[i] = cam[i-1]+vel[i]
    return cam

cx = follow(tx, w, dt); cy = follow(ty, h, dt)

# zoom: wider default + predictive widen on bursts
def z_needed(cx,cy):
    dx = np.minimum(cx, w-cx) - MX
    dy = np.minimum(cy, h-cy) - MY
    dx = np.clip(dx, 1, None); dy = np.clip(dy, 1, None)
    safety=1.05
    zx=(h*9/16)/(safety*2*dx); zy=(h)/(safety*2*dy)
    return np.maximum(1.0, np.maximum(zx,zy))

z_base = 1.02
z_need = z_needed(cx,cy)
z_pred = 1.00 + 0.20*np.clip((spd-120)/220,0,1)   # stronger widen on speed
z = np.minimum(ZMAX, np.maximum.reduce([np.full_like(z_need,z_base), z_need, z_pred]))

# runtime keep-in around true ball
for i in range(T):
    cw = min((h*9/16)/z[i], w); ch = min(h/z[i], h)
    cx[i] = np.clip(cx[i], cw/2, w-cw/2)
    cy[i] = np.clip(cy[i], ch/2, h-ch/2)
    left=cx[i]-cw/2; right=cx[i]+cw/2; top=cy[i]-ch/2; bot=cy[i]+ch/2
    bx,by = X[i],Y[i]
    need=False
    if bx-left<EDGE:   cx[i]=min(w-cw/2, bx-(cw/2-EDGE)); need=True
    if right-bx<EDGE:  cx[i]=max(cw/2,   bx+(cw/2-EDGE)); need=True
    if by-top<EDGE:    cy[i]=min(h-ch/2, by-(ch/2-EDGE)); need=True
    if bot-by<EDGE:    cy[i]=max(ch/2,   by+(ch/2-EDGE)); need=True
    if need:
        z[i]=min(ZMAX, max(z[i], z_needed(cx[i],cy[i])*1.03))

# zoom dynamics: much faster widen, modest tighten
for i in range(1,T):
    dz=z[i]-z[i-1]
    if dz >  0.020*SPEED: z[i]=z[i-1]+0.020*SPEED
    if dz < -0.090*SPEED: z[i]=z[i-1]-0.090*SPEED

# tiny smoothing to remove micro jitter only
cx=ema(cx,0.10); cy=ema(cy,0.10)

# Compact expressions (poly with pow(); one piece per third)
def fit_pw(arr, deg_c=5):
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

cx_expr = fit_pw(cx,5)
cy_expr = fit_pw(cy,5)
z_expr  = "clip(" + fit_pw(z,deg_c=3) + f",1.00,{ZMAX:.2f})"
with open(out_ps1,"w",encoding="utf-8") as f:
    f.write(f"$cxExpr='={cx_expr}'`n$cyExpr='={cy_expr}'`n$zExpr='={z_expr}'`n")
