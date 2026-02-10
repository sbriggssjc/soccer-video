import sys, csv, numpy as np

if len(sys.argv)<3: raise SystemExit("usage: plan_camera.py <track_csv> <out_ps1>")
in_csv, out_ps1 = sys.argv[1], sys.argv[2]

N,cx,cy,conf,speed=[],[],[],[],[]
w=h=fps=None
with open(in_csv,"r",newline="") as f:
    rd=csv.DictReader(f)
    for row in rd:
        N.append(int(row["n"]))
        cx.append(float(row["cx"])); cy.append(float(row["cy"]))
        conf.append(float(row["conf"])); speed.append(float(row["speed"]))
        w = int(row["w"]); h = int(row["h"]); fps = float(row["fps"])
N=np.array(N); cx=np.array(cx); cy=np.array(cy); conf=np.array(conf); speed=np.array(speed)

def ema(a, alpha):
    o=np.copy(a)
    for i in range(1,len(a)): o[i]=alpha*a[i]+(1-alpha)*o[i-1]
    return o

# Fill short gaps
for i in range(1,len(cx)):
    if conf[i] < 0.2:
        cx[i]=cx[i-1]; cy[i]=cy[i-1]

# Smoother base path
cx = ema(ema(cx,0.45),0.26)
cy = ema(ema(cy,0.45),0.26)

# Longer lead so we start with the ball: 0.40 s + warmer ramp (28 f)
lead = int(round(fps*0.40))
if lead>0 and lead<len(cx):
    cx = np.concatenate([cx[lead:], np.repeat(cx[-1], lead)])
    cy = np.concatenate([cy[lead:], np.repeat(cy[-1], lead)])
    N  = np.arange(len(cx))
warm = min(28,len(cx)-1)
aw = np.linspace(0,1,warm)
cx[:warm] = cx[:warm]*(aw) + (w/2.0)*(1-aw)
cy[:warm] = cy[:warm]*(aw) + (h/2.0)*(1-aw)

# --- Jerk + speed limits on the virtual head ---
def rate_limit(x, dx_max):
    y = np.copy(x)
    for i in range(1,len(y)):
        d = y[i]-y[i-1]
        if d >  dx_max: y[i] = y[i-1] + dx_max
        if d < -dx_max: y[i] = y[i-1] - dx_max
    return y

def jerk_limit(p, dv_max=8.0, da_max=0.6):  # stricter than before
    v = np.zeros_like(p); a = np.zeros_like(p)
    out = np.copy(p)
    # velocity cap
    for i in range(1,len(p)):
        v[i] = np.clip(out[i]-out[i-1], -dv_max, dv_max)
        out[i] = out[i-1] + v[i]
    # acceleration cap
    for i in range(2,len(p)):
        a[i] = np.clip((out[i]-2*out[i-1]+out[i-2]), -da_max, da_max)
        out[i] = 2*out[i-1]-out[i-2] + a[i]
    return out

# Apply jerk + a hard per-frame pan clamp (prevents little snaps)
cx = jerk_limit(rate_limit(cx, dx_max=7.5))
cy = jerk_limit(rate_limit(cy, dx_max=7.5))

# --- Zoom plan: more context + confidence/speed widen ---
spd = ema(speed,0.35)
mx_base, my_base = 170.0, 200.0      # wider margins so ball stays in frame during bursts
wide_boost = np.clip((spd-240.0)/220.0, 0, 1) + np.clip((0.35-conf)/0.35, 0, 1)*0.8
mx = mx_base*(1.0+0.55*wide_boost)   # was 0.35 ? 0.55
my = my_base*(1.0+0.55*wide_boost)

dx = np.minimum(cx, w-cx) - mx
dy = np.minimum(cy, h-cy) - my
dx = np.clip(dx,1,None); dy=np.clip(dy,1,None)
safety=1.08                               # slightly higher overscan to prevent edge hits
z_need_x = (h*9/16)/(safety*2*dx)
z_need_y = (h)/(safety*2*dy)
z = np.maximum(1.0, np.maximum(z_need_x, z_need_y))
z = np.minimum(1.50, 0.80*z + 0.20)       # cap lower than before for steadier look
z = ema(z, 0.32)

# Hard per-frame zoom-rate cap to remove “pumping”
def zoom_rate_limit(z, dz_max=0.018):     # ~1.8% per frame at source fps
    out = np.copy(z)
    for i in range(1,len(out)):
        d = out[i]-out[i-1]
        if d >  dz_max: out[i] = out[i-1] + dz_max
        if d < -dz_max: out[i] = out[i-1] - dz_max
    return out
z = zoom_rate_limit(z)

# Piecewise expressions with pow(n,k); safe tails so ffmpeg never sees empty args
def piecewise_expr(n, y, deg, S=4):
    idx = np.linspace(0, len(n)-1, S+1, dtype=int)
    parts=[]
    for s in range(S):
        lo = idx[s]
        hi = idx[s+1]-1 if s<S-1 else idx[s+1]-1
        nn = n[lo:hi+1].astype(float); yy = y[lo:hi+1].astype(float)
        d = min(deg, max(1,len(nn)-1))
        P = np.polyfit(nn, yy, d)
        terms=[]; powmax=len(P)-1
        for i,a in enumerate(P):
            k = powmax-i; coef = f"{a:.10g}"
            if k==0: terms.append(f"({coef})")
            elif k==1: terms.append(f"({coef})*n")
            else: terms.append(f"({coef})*pow(n,{k})")
        expr = "(" + "+".join(terms) + ")"
        cond = f"between(n,{lo},{hi})"
        parts.append(f"if({cond},{expr},")
    tail="(0)"; return "".join(parts)+tail+")"*S

cx_expr = piecewise_expr(N, cx, 3, S=4)
cy_expr = piecewise_expr(N, cy, 3, S=4)
z_expr  = "clip(" + piecewise_expr(N, z, 2, S=4) + ",1.0,1.50)"

with open(out_ps1,"w",encoding="utf-8") as f:
    f.write(f"$cxExpr = '={cx_expr}'\n")
    f.write(f"$cyExpr = '={cy_expr}'\n")
    f.write(f"$zExpr  = '={z_expr}'\n")
