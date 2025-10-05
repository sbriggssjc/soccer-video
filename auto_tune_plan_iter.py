import sys, csv, numpy as np
try:
    from numpy import RankWarning
except Exception:
    class RankWarning(UserWarning): pass
import warnings
warnings.filterwarnings("ignore", category=RankWarning)

if len(sys.argv)<3:
    raise SystemExit("usage: auto_tune_plan_iter.py <ball_csv> <out_ps1vars>")
ball_csv, out_ps1 = sys.argv[1], sys.argv[2]

# --- load ---
N=[]; bx=[]; by=[]; conf=[]; W=H=FPS=None
with open(ball_csv,"r",newline="") as f:
    rd=csv.DictReader(f)
    for r in rd:
        N.append(int(r["n"]))
        bx.append(float(r["cx"])); by.append(float(r["cy"])); conf.append(float(r["conf"]))
        W=int(r["w"]); H=int(r["h"]); FPS=float(r["fps"])
N=np.array(N); bx=np.array(bx); by=np.array(by); conf=np.array(conf)
if len(N)==0: raise SystemExit("Empty CSV")
dt = 1.0/max(FPS,1.0)

def ema(a,alpha):
    o=a.copy()
    for i in range(1,len(a)):
        o[i]=alpha*a[i]+(1-alpha)*o[i-1]
    return o

def slew_follow(target, current, dz, gain, slew):
    out = current.copy()
    for i in range(1,len(current)):
        err = target[i]-out[i-1]
        step = gain*err*(1.8 if abs(err)>dz else 1.0)
        if step >  slew: step =  slew
        if step < -slew: step = -slew
        out[i] = out[i-1] + step
    return out

def make_center(LA_s, ACC, GAIN, SLEW, DZx, DZy, s1=0.26, s2=0.15):
    cx_s, cy_s = ema(bx, s1), ema(by, s1); cx_s, cy_s = ema(cx_s, s2), ema(cy_s, s2)
    vx = np.gradient(cx_s, dt); vy = np.gradient(cy_s, dt)
    ax = np.gradient(vx, dt);   ay = np.gradient(vy, dt)
    A_CLAMP = 3300.0
    ax = np.clip(ax,-A_CLAMP,A_CLAMP); ay = np.clip(ay,-A_CLAMP, A_CLAMP)
    cx_pred = cx_s + vx*LA_s + 0.5*ACC*ax*(LA_s**2)
    cy_pred = cy_s + vy*LA_s + 0.5*ACC*ay*(LA_s**2)
    warm = int(round(0.05*FPS))
    cx_t = np.where(N<warm, cx_s, cx_pred)
    cy_t = np.where(N<warm, cy_s, cy_pred)
    cx_ff = slew_follow(cx_t, cx_s, DZx, GAIN, SLEW)
    cy_ff = slew_follow(cy_t, cy_s, DZy, GAIN, SLEW)
    return cx_ff, cy_ff

def containment_zoom_needed(cx, cy, mx, my, safety):
    dx_ball = np.abs(bx - cx); dy_ball = np.abs(by - cy)
    z_need_ball_x = (H*9/16)/(safety*2*(dx_ball+mx).clip(min=1))
    z_need_ball_y = (H)/(safety*2*(dy_ball+my).clip(min=1))
    return np.maximum(z_need_ball_x, z_need_ball_y)

def score_and_zoom(cx, cy, base_mx, base_my, safety, zcap=3.0):
    vx = np.gradient(bx, dt); vy = np.gradient(by, dt)
    vmag = np.hypot(vx,vy) + 1e-6; ux, uy = vx/vmag, vy/vmag
    ex = bx - cx; ey = by - cy
    lead = ex*ux + ey*uy
    mae  = np.mean(np.hypot(ex,ey))
    p95  = np.percentile(np.maximum(0.0, lead), 95.0)

    spd = vmag; vsp = np.clip(spd/1000.0, 0, 1)
    mx = base_mx + 120.0*vsp + 110.0*(conf<0.25)
    my = base_my + 140.0*vsp + 110.0*(conf<0.25)

    # edge-based need
    dx_edge = np.minimum(cx, W-cx) - mx
    dy_edge = np.minimum(cy, H-cy) - my
    dx_edge = np.clip(dx_edge,1,None); dy_edge = np.clip(dy_edge,1,None)
    z_need_edge_x = (H*9/16)/(safety*2*dx_edge)
    z_need_edge_y = (H)/(safety*2*dy_edge)

    # ball containment need
    z_need_ball = containment_zoom_needed(cx, cy, mx, my, safety)

    z_need = np.maximum.reduce([np.ones_like(dx_edge),
                                z_need_edge_x, z_need_edge_y,
                                z_need_ball])

    def ema_vec(a,alpha):
        o=a.copy()
        for i in range(1,len(a)): o[i]=alpha*a[i]+(1-alpha)*o[i-1]
        return o
    z_plan = np.minimum(np.maximum(1.0, 0.78*z_need + 0.36), zcap)
    z_plan = ema_vec(z_plan, 0.10)
    excess = np.maximum(z_need - z_plan, 0.0)
    alpha  = np.clip(excess/0.038, 0, 1)
    z_soft = z_plan*(1-alpha) + z_need*alpha
    z_final= np.minimum(np.maximum(z_soft, np.minimum(zcap, z_need*1.17)), zcap)

    viol = float(np.any(z_need > zcap))
    edge_pen = np.mean((cx-mx<0)+(W-cx-mx<0)+(cy-my<0)+(H-cy-my<0))

    score = 4.0*p95 + 1.0*mae + 400.0*viol + 30.0*edge_pen
    return score, p95, mae, viol, z_final, mx, my, z_need_ball

# iterative search
TARGET_P95 = 6.0
TARGET_MAE = 10.0
MAX_ITERS  = 7
LA_rng   = [2.0, 2.4, 2.8, 3.0]
ACC_rng  = [0.15, 0.25, 0.35]
GAIN_rng = [0.70, 0.90, 1.10]
SLEW_rng = [100.0, 130.0, 160.0, 180.0]
DZx, DZy = 80.0, 95.0
MX_rng   = [160.0, 180.0, 200.0]
MY_rng   = [190.0, 210.0, 230.0]
SAF_rng  = [1.06, 1.08, 1.10]
ZCAP     = 3.0

best=None
for it in range(MAX_ITERS):
    for LA_s in LA_rng:
      for ACC in ACC_rng:
        for GAIN in GAIN_rng:
          for SLEW in SLEW_rng:
            for base_mx in MX_rng:
              for base_my in MY_rng:
                for safety in SAF_rng:
                    cx, cy = make_center(LA_s, ACC, GAIN, SLEW, DZx, DZy)
                    # ---- panic-center refinement (single pass) ----
                    spd = np.hypot(np.gradient(bx,dt), np.gradient(by,dt))
                    vsp = np.clip(spd/1000.0,0,1)
                    mx = base_mx + 120.0*vsp + 110.0*(conf<0.25)
                    my = base_my + 140.0*vsp + 110.0*(conf<0.25)
                    z_need_ball = containment_zoom_needed(cx, cy, mx, my, safety)
                    panic = z_need_ball > (0.92*ZCAP)
                    if np.any(panic):
                        # pull center toward ball with high slew when containment under pressure
                        PC_GAIN = 1.6
                        PC_SLEW = 220.0  # px/frame cap during panic
                        cx2 = cx.copy(); cy2 = cy.copy()
                        for i in range(1,len(cx2)):
                            if panic[i]:
                                ex = bx[i]-cx2[i-1]; ey = by[i]-cy2[i-1]
                                sx = np.clip(PC_GAIN*ex, -PC_SLEW, PC_SLEW)
                                sy = np.clip(PC_GAIN*ey, -PC_SLEW, PC_SLEW)
                                cx2[i] = cx2[i-1] + sx
                                cy2[i] = cy2[i-1] + sy
                            else:
                                cx2[i] = cx2[i]
                                cy2[i] = cy2[i]
                        cx, cy = cx2, cy2

                    score, p95, mae, viol, zf, mx, my, _ = score_and_zoom(cx, cy, base_mx, base_my, safety, zcap=ZCAP)
                    if (best is None) or (score < best[0]):
                        best = [score, p95, mae, viol, LA_s, ACC, GAIN, SLEW, base_mx, base_my, safety, cx, cy, zf]
    if best[1] <= TARGET_P95 and best[2] <= TARGET_MAE and best[3] == 0.0:
        break
    # recenter ranges around best + widen
    _, _, _, _, LA_s, ACC, GAIN, SLEW, base_mx, base_my, safety, _, _, _ = best
    def span(v, lo, hi, step):
        vals=set([v])
        for k in [1,2]:
            vals.add(max(lo, v - k*step)); vals.add(min(hi, v + k*step))
        return sorted(vals)
    LA_rng   = span(LA_s, 1.6, 3.2, 0.2)
    ACC_rng  = span(ACC,  0.05, 0.45, 0.05)
    GAIN_rng = span(GAIN, 0.60, 1.40, 0.10)
    SLEW_rng = span(SLEW, 80.0, 220.0, 10.0)
    MX_rng   = span(base_mx, 120.0, 240.0, 20.0)
    MY_rng   = span(base_my, 150.0, 260.0, 20.0)
    SAF_rng  = span(safety, 1.04, 1.12, 0.02)

# unpack best
_, p95, mae, viol, LA_s, ACC, GAIN, SLEW, base_mx, base_my, safety, cx_ff, cy_ff, z_final = best

# piecewise -> ffmpeg expr
def piecewise(n,y,seg=12,deg=3):
    S=[]; i=0
    while i<len(n):
        j=min(i+seg-1, len(n)-1)
        nn=n[i:j+1].astype(float); yy=y[i:j+1].astype(float)
        d=min(deg, len(nn)-1) if len(nn)>1 else 1
        co=np.polyfit(nn, yy, d)
        S.append((int(n[i]), int(n[j]), co)); i=j+1
    return S

def pexpr(co):
    d=len(co)-1; t=[]
    for k,a in enumerate(co):
        p=d-k; c=f"{a:.10g}"
        t.append(f"({c})" if p==0 else (f"({c})*n" if p==1 else f"({c})*n^{p}"))
    return "(" + "+".join(t) + ")"

def seg_if(segs, fallback):
    return "".join([f"if(between(n,{a},{b}),{pexpr(co)}," for a,b,co in segs]) + fallback + (")"*len(segs))

cx_if = seg_if(piecewise(N,cx_ff,seg=12,deg=3), "in_w/2")
cy_if = seg_if(piecewise(N,cy_ff,seg=12,deg=3), "in_h/2")
z_if  = seg_if(piecewise(N,z_final,seg=12,deg=2), "1")

with open(out_ps1,"w",encoding="utf-8") as f:
    f.write(f"$cxExpr = '={cx_if}'\n")
    f.write(f"$cyExpr = '=={cy_if}'.replace('==','=')\n")
    f.write(f"$zExpr  = '=clip({z_if},1.0,3.00)'\n")
    f.write(f"$Safety = {safety:.3f}\n")
    f.write(f"# tuned: p95_lead={p95:.2f}, mae={mae:.2f}, viol={viol:.4f}, LA={LA_s}, ACC={ACC}, GAIN={GAIN}, SLEW={SLEW}, mx={base_mx}, my={base_my}`n")
