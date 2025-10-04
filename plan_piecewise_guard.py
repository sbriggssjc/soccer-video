import sys, csv, math, numpy as np
if len(sys.argv)<3: raise SystemExit("usage: plan_piecewise_guard.py track.csv out.ps1vars")
in_csv, out_ps1 = sys.argv[1], sys.argv[2]

# --- Load track ---
N=[]; CX=[]; CY=[]; CONF=[]
w=h=fps=None
with open(in_csv,"r",newline="") as f:
    rd=csv.DictReader(f)
    for r in rd:
        N.append(int(r["n"])); CX.append(float(r["cx"])); CY.append(float(r["cy"])); CONF.append(float(r["conf"]))
        w = int(r["w"]); h = int(r["h"]); fps = float(r["fps"])
N=np.array(N); CX=np.array(CX); CY=np.array(CY); CONF=np.array(CONF)
T=len(N); fps = fps if fps and fps>1e-6 else 24.0; dt=1.0/fps

# --- Kalman (CV) smoothing ---
def kcv(cx,cy,conf,dt,q=7.0,r0=8.0):
    X=np.zeros(4); P=np.eye(4)*1e3
    F=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],float)
    Q=np.array([[dt**4/4,0,dt**3/2,0],[0,dt**4/4,0,dt**3/2],[dt**3/2,0,dt**2,0],[0,dt**3/2,0,dt**2]],float)*q
    H=np.array([[1,0,0,0],[0,1,0,0]],float)
    xs=np.zeros(T); ys=np.zeros(T); vx=np.zeros(T); vy=np.zeros(T)
    X[:2]=[cx[0],cy[0]]
    for i in range(T):
        X=F@X; P=F@P@F.T+Q
        c=max(0.05,min(1.0,conf[i])); R=np.eye(2)*(r0/c)
        S=H@P@H.T+R; K=P@H.T@np.linalg.inv(S)
        z=np.array([cx[i],cy[i]])
        X = X + K@(z-H@X); P=(np.eye(4)-K@H)@P
        xs[i],ys[i],vx[i],vy[i]=X
    return xs,ys,vx,vy
xs,ys,vx,vy = kcv(CX,CY,CONF,dt)

# --- Predictive target ---
spd = np.hypot(vx,vy)
heading = np.arctan2(vy, vx+1e-6); dhead = np.diff(np.unwrap(heading)); dhead = np.concatenate([[0.0],dhead])
curv = np.abs(dhead)/(dt+1e-9)
lead_s = 0.35 + np.clip((spd-160)/240,0,1)*0.35 - np.clip(curv/6,0,1)*0.20
lead_s = np.clip(lead_s, 0.20, 0.80); kfrm = np.round(lead_s/dt).astype(int)
tx = np.clip(xs + vx*(kfrm*dt), 1, w-2)
ty = np.clip(ys + vy*(kfrm*dt), 1, h-2)

# --- Jerk-limited follow with gain schedule ---
def limits(err_norm, spd):
    v = 12 + 32*np.clip(err_norm,0,1.5) + 18*np.clip((spd-160)/260,0,1)
    a = 0.7 + 1.6*np.clip(err_norm,0,1.5) + 1.0*np.clip((spd-160)/260,0,1)
    return v, a
def scurve(target, W, dt):
    cam=np.zeros_like(target); vel=np.zeros_like(target); cam[0]=target[0]
    for i in range(1,len(target)):
        e = target[i]-cam[i-1]; en = abs(e)/(0.5*W)
        vlim, alim = limits(en, spd[i])
        flip = np.clip(curv[i]/5.0, 0.0, 1.0); vlim *= (1-0.50*flip)
        kp=0.20+0.22*np.clip(en,0,1.2); kv=0.12
        a_cmd = kp*e - kv*vel[i-1]
        if np.sign(a_cmd)==np.sign(e): a_cmd=np.clip(a_cmd,-alim,+alim*1.35)
        else:                           a_cmd=np.clip(a_cmd,-alim*0.9,+alim)
        if i>=2:
            jerk = 0.7 + 1.0*np.clip(en,0,1.4)
            da = np.clip(a_cmd - (vel[i-1]-vel[i-2]), -jerk, +jerk)
            a_cmd = (vel[i-1]-vel[i-2]) + da
        vel[i]=np.clip(vel[i-1]+a_cmd, -vlim, +vlim)
        cam[i]=cam[i-1]+vel[i]
    return cam
cx=scurve(tx, w, dt); cy=scurve(ty, h, dt)

# --- Base zoom from margins ---
mx,my = 170.0, 210.0
def z_need(cx,cy):
    dx = np.minimum(cx, w-cx) - mx; dy = np.minimum(cy, h-cy) - my
    dx = np.clip(dx, 1, None); dy = np.clip(dy,1,None); safety=1.06
    zx=(h*9/16)/(safety*2*dx); zy=(h)/(safety*2*dy)
    return np.maximum(1.0, np.maximum(zx,zy))
zn = z_need(cx,cy)

# --- EMERGENCY INCLUSION PASS (guarantee ball inside crop) ---
# If ball would be within edge margin in simulated crop, recenter a bit toward ball and widen zoom until safe.
edge_px = 24.0  # keep ball at least this many pixels from crop edge
for i in range(T):
    z = min(1.60, max(zn[i], 1.05))   # allow to widen up to 1.60
    # crop size
    cw = min((h*9/16)/z, w); ch = min(h/z, h)
    # ball position relative to crop center
    bx, by = CX[i], CY[i]
    # desired camera center (current plan)
    cx_i, cy_i = cx[i], cy[i]
    # Clamp center so crop stays inside frame:
    cx_i = np.clip(cx_i, cw/2, w - cw/2)
    cy_i = np.clip(cy_i, ch/2, h - ch/2)
    # Check margins; if ball is too close to an edge, slide center toward ball and widen z a bit
    left   = cx_i - cw/2; right = cx_i + cw/2
    top    = cy_i - ch/2; bot   = cy_i + ch/2
    need_wider = False
    if bx - left   < edge_px: cx_i = min(w - cw/2, bx - (cw/2 - edge_px)); need_wider = True
    if right - bx  < edge_px: cx_i = max(cw/2,      bx + (cw/2 - edge_px)); need_wider = True
    if by - top    < edge_px: cy_i = min(h - ch/2, by - (ch/2 - edge_px)); need_wider = True
    if bot  - by   < edge_px: cy_i = max(ch/2,      by + (ch/2 - edge_px)); need_wider = True
    if need_wider:
        # widen just enough to respect edge_px
        dxe = max(edge_px - (bx - (cx_i - cw/2)), edge_px - ((cx_i + cw/2) - bx), 0)
        dye = max(edge_px - (by - (cy_i - ch/2)), edge_px - ((cy_i + ch/2) - by), 0)
        if dxe>0 or dye>0:
            # approximate additional zoom need
            zx = (h*9/16)/( (cw/2 - (edge_px))*2 )
            zy = (h)/( (ch/2 - (edge_px))*2 )
            z = min(1.60, max(z, zx, zy))
            cw = min((h*9/16)/z, w); ch = min(h/z, h)
            cx_i = np.clip(bx, cw/2, w - cw/2)
            cy_i = np.clip(by, ch/2, h - ch/2)
    cx[i], cy[i], zn[i] = cx_i, cy_i, z

# gentle finish smooth on center; constrain zoom tighten speed
def ema(a,alpha):
    o=np.copy(a); 
    for i in range(1,len(a)): o[i] = alpha*a[i] + (1-alpha)*o[i-1]
    return o
cx=ema(cx,0.22); cy=ema(cy,0.22)
z = np.maximum(zn, z_need(cx,cy))
for i in range(1,T):
    dz=z[i]-z[i-1]
    if dz> 0.012: z[i]=z[i-1]+0.012  # limit quick tighten
    if dz<-0.045: z[i]=z[i-1]-0.045  # allow fast widen

# --- Piecewise fits (3 segments) to avoid mid-clip drift ---
# breakpoints at ~1/3 and ~2/3 of frames
b0=0; b1=int(0.33*(T-1)); b2=int(0.67*(T-1)); b3=T-1
def fit_seg(n0,n1,arr,deg):
    n = np.arange(n0,n1+1); u=(n-n0)/max(1,(n1-n0))
    P = np.polyfit(u, arr[n0:n1+1], deg); d=len(P)-1; terms=[]
    U="((n-{n0})/{span})".format(n0=n0, span=max(1,(n1-n0)))
    for i,a in enumerate(P):
        k=d-i; c=f"{a:.10g}"
        if   k==0: terms.append(f"({c})")
        elif k==1: terms.append(f"({c})*{U}")
        else:      terms.append(f"({c})*pow({U},{k})")
    return "(" + "+".join(terms) + ")", n0, n1
def pw(exprs, default):
    # build if(between(n,a,b), expr, if(...))
    out = default
    for e,a,b in reversed(exprs):
        out = f"if(between(n,{a},{b}), {e}, {out})"
    return out

cx_e1,a1,b1 = fit_seg(b0,b1,cx,5)
cx_e2,a2,b2 = fit_seg(b1,b2,cx,5)
cx_e3,a3,b3 = fit_seg(b2,b3,cx,5)
cx_expr = pw([(cx_e1,a1,b1),(cx_e2,a2,b2),(cx_e3,a3,b3)], f"({cx[-1]:.6g})")

cy_e1,a1,b1 = fit_seg(b0,b1,cy,5)
cy_e2,a2,b2 = fit_seg(b1,b2,cy,5)
cy_e3,a3,b3 = fit_seg(b2,b3,cy,5)
cy_expr = pw([(cy_e1,a1,b1),(cy_e2,a2,b2),(cy_e3,a3,b3)], f"({cy[-1]:.6g})")

z_e1,a1,b1 = fit_seg(b0,b1,z,3)
z_e2,a2,b2 = fit_seg(b1,b2,z,3)
z_e3,a3,b3 = fit_seg(b2,b3,z,3)
z_expr = "clip(" + pw([(z_e1,a1,b1),(z_e2,a2,b2),(z_e3,a3,b3)], f"({z[-1]:.6g})") + ",1.0,1.60)"

with open(out_ps1,"w",encoding="utf-8") as f:
    f.write(f"$cxExpr = '={cx_expr}'\n")
    f.write(f"$cyExpr = '={cy_expr}'\n")
    f.write(f"$zExpr  = '={z_expr}'\n")
