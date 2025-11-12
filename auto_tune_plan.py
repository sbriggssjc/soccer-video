import sys, csv, numpy as np
try:
    from numpy import RankWarning
except Exception:
    class RankWarning(UserWarning): pass
import warnings
warnings.filterwarnings("ignore", category=RankWarning)

if len(sys.argv)<3:
    raise SystemExit("usage: auto_tune_plan.py <ball_csv> <out_ps1vars>")
ball_csv, out_ps1 = sys.argv[1], sys.argv[2]

# ---- load ball path ----
N=[]; cx=[]; cy=[]; conf=[]; W=H=FPS=None
with open(ball_csv,"r",newline="") as f:
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

def slew_follow(target, current, dz, gain, slew):
    out = current.copy()
    for i in range(1,len(current)):
        err = target[i]-out[i-1]
        step = gain*err*(1.8 if abs(err)>dz else 1.0)
        if step >  slew: step =  slew
        if step < -slew: step = -slew
        out[i] = out[i-1] + step
    return out

# error metrics
def metrics(center_x, center_y, base_mx, base_my, safety):
    # distance / "lateness" along velocity direction
    vx = np.gradient(cx, dt); vy = np.gradient(cy, dt)
    vnorm = np.hypot(vx,vy)+1e-6
    ux, uy = vx/vnorm, vy/vnorm  # unit velocity
    dx = cx - center_x
    dy = cy - center_y
    # signed lead error: + means ball is ahead of center in motion direction
    lead = dx*ux + dy*uy
    mae  = np.mean(np.hypot(dx,dy))
    p95_lead = np.percentile(np.maximum(0.0, lead), 95.0)
    # penalty for crop hitting edges (widening need): if too tight, zoom skyrockets
    # approximate needed zoom from margins to ensure portrait crop exists
    mx = base_mx; my = base_my
    left  = center_x - mx; right = (W-center_x) - mx
    top   = center_y - my; bottom= (H-center_y) - my
    edge_pen = np.mean((left<0)+(right<0)+(top<0)+(bottom<0))
    # weighted score
    score = 3.0*p95_lead + 1.0*mae + 50.0*edge_pen
    return score, p95_lead, mae

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

# search spaces (adjust if needed)
LA_space     = [1.8, 2.0, 2.2, 2.4]
ACC_space    = [0.15, 0.25, 0.35]
GAIN_space   = [0.55, 0.65, 0.75]
SLEW_space   = [50.0, 60.0, 70.0]
MX_space     = [160.0, 180.0]   # base_mx
MY_space     = [190.0, 210.0]   # base_my
SAF_space    = [1.06, 1.08, 1.10]

best = None

for LA_s in LA_space:
  for ACC in ACC_space:
    for GAIN in GAIN_space:
      for SLEW in SLEW_space:
        for base_mx in MX_space:
          for base_my in MY_space:
            # smoothing
            cx_s, cy_s = ema(cx,0.26), ema(cy,0.26)
            cx_s, cy_s = ema(cx_s,0.15), ema(cy_s,0.15)

            # v/a + prediction
            vx = np.gradient(cx_s, dt); vy = np.gradient(cy_s, dt)
            ax = np.gradient(vx, dt);   ay = np.gradient(vy, dt)
            A_CLAMP = 2600.0
            ax = np.clip(ax, -A_CLAMP, A_CLAMP); ay = np.clip(ay, -A_CLAMP, A_CLAMP)
            cx_pred = cx_s + vx*LA_s + 0.5*ACC*ax*(LA_s**2)
            cy_pred = cy_s + vy*LA_s + 0.5*ACC*ay*(LA_s**2)

            # warm-start blending
            warm = int(round(0.05*FPS))
            cx_t = np.where(N<warm, cx_s, cx_pred)
            cy_t = np.where(N<warm, cy_s, cy_pred)

            # dead-zone + slew
            DZ_X, DZ_Y = 85.0, 100.0
            cx_ff = slew_follow(cx_t, cx_s, DZ_X, GAIN, SLEW)
            cy_ff = slew_follow(cy_t, cy_s, DZ_Y, GAIN, SLEW)

            # evaluate at median safety (we’ll pick best safety later for zoom expr)
            score, p95_lead, mae = metrics(cx_ff, cy_ff, base_mx, base_my, 1.08)
            if (best is None) or (score < best[0]):
                best = (score, p95_lead, mae, LA_s, ACC, GAIN, SLEW, base_mx, base_my, cx_ff, cy_ff)

# pick a safety by checking zoom feasibility (light heuristic)
_, _, _, LA_s, ACC, GAIN, SLEW, base_mx, base_my, cx_ff, cy_ff = best
safety_candidates = [1.06, 1.08, 1.10]
def pick_safety():
    vx = np.gradient(cx_ff, dt); vy = np.gradient(cy_ff, dt)
    spd = np.hypot(vx,vy); vnorm = np.clip(spd/1000.0, 0, 1)
    for saf in safety_candidates:
        dxm = np.minimum(cx_ff, W-cx_ff) - (base_mx + 120.0*vnorm + 110.0*(np.array(conf)<0.25))
        dym = np.minimum(cy_ff, H-cy_ff) - (base_my + 140.0*vnorm + 110.0*(np.array(conf)<0.25))
        if np.all(dxm>1.0) and np.all(dym>1.0):
            return saf
    return 1.10
safety = pick_safety()

# compute zoom plan at chosen safety
vx = np.gradient(cx_ff, dt); vy = np.gradient(cy_ff, dt)
spd = np.hypot(vx,vy); vnorm = np.clip(spd/1000.0, 0, 1)
mx = base_mx + 120.0*vnorm + 110.0*(np.array(conf)<0.25)
my = base_my + 140.0*vnorm + 110.0*(np.array(conf)<0.25)
dx = np.minimum(cx_ff, W-cx_ff) - mx
dy = np.minimum(cy_ff, H-cy_ff) - my
dx = np.clip(dx, 1, None); dy = np.clip(dy, 1, None)
z_need_x = (H*9/16)/(safety*2*dx)
z_need_y = (H)/(safety*2*dy)
z_needed = np.maximum(1.0, np.maximum(z_need_x, z_need_y))
z_plan = np.minimum(np.maximum(1.0, 0.78*z_needed + 0.36), 2.60)
def ema_vec(a,alpha):
    o=a.copy()
    for i in range(1,len(a)): o[i]=alpha*a[i]+(1-alpha)*o[i-1]
    return o
z_plan = ema_vec(z_plan, 0.10)
excess = np.maximum(z_needed - z_plan, 0.0)
alpha  = np.clip(excess/0.038, 0, 1)
z_soft = z_plan*(1-alpha) + z_needed*alpha
z_final= np.minimum(np.maximum(z_soft, np.minimum(2.60, z_needed*1.17)), 2.60)

# piecewise -> ffmpeg expr
cx_if = seg_if(piecewise(N,cx_ff,seg=12,deg=3), "in_w/2")
cy_if = seg_if(piecewise(N,cy_ff,seg=12,deg=3), "in_h/2")
z_if  = seg_if(piecewise(N,z_final,seg=12,deg=2), "1")

with open(out_ps1,"w",encoding="utf-8") as f:
    f.write(f"$cxExpr = '={cx_if}'\n")
    f.write(f"$cyExpr = '=={cy_if}'.replace('==','=')\n")
    f.write(f"$zExpr  = '=clip({z_if},1.0,2.60)'\n")
    f.write(f"$Safety = {safety}\n")
