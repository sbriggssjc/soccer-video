import sys, csv, numpy as np

if len(sys.argv)<3: raise SystemExit("usage: fit_piecewise_expr.py <in_csv> <out_ps1>")
in_csv, out_ps1 = sys.argv[1], sys.argv[2]

# load
N,cx,cy,conf=[],[],[],[]
w=h=fps=None
with open(in_csv,"r",newline="") as f:
    rd=csv.DictReader(f)
    for row in rd:
        N.append(int(row["n"]))
        cx.append(float(row["cx"])); cy.append(float(row["cy"])); conf.append(float(row["conf"]))
        w = int(row["w"]); h = int(row["h"]); fps = float(row["fps"])
N=np.array(N); cx=np.array(cx); cy=np.array(cy); conf=np.array(conf)

if len(N)<8: raise SystemExit("Too few samples")

# hold small gaps
for i in range(1,len(cx)):
    if conf[i] < 0.2:
        cx[i] = cx[i-1]; cy[i] = cy[i-1]

# EMA smoothing
def ema(a, alpha):
    o=np.copy(a)
    for i in range(1,len(a)): o[i]=alpha*a[i]+(1-alpha)*o[i-1]
    return o
cx = ema(ema(cx,0.35),0.20)
cy = ema(ema(cy,0.35),0.20)

# lead ~0.25s
lead = int(round(fps*0.25))
if lead>0:
    cx = np.concatenate([cx[lead:], np.repeat(cx[-1], lead)])
    cy = np.concatenate([cy[lead:], np.repeat(cy[-1], lead)])
    N  = np.arange(len(cx))

# zoom from margins, soft bias wide
mx,my = 120.0,150.0
dx = np.minimum(cx, w-cx) - mx
dy = np.minimum(cy, h-cy) - my
dx = np.clip(dx, 1, None); dy=np.clip(dy,1,None)
safety=1.05
z_need_x = (h*9/16)/(safety*2*dx)
z_need_y = (h)/(safety*2*dy)
z = np.maximum(1.0, np.maximum(z_need_x, z_need_y))
z = np.minimum(1.55, 0.75*z + 0.25)
z = ema(z,0.25)

# piecewise segments (4 roughly equal)
S = 4
indices = np.linspace(0, len(N)-1, S+1, dtype=int)

def fit_poly_piece(n, y, lo, hi, deg):
    sl = slice(lo, hi+1)
    nn = n[sl].astype(float); yy = y[sl].astype(float)
    if len(nn) <= deg: deg = max(1, min(deg, len(nn)-1))
    P = np.polyfit(nn, yy, deg)
    # to ffmpeg pow(n,k) expr
    terms=[]
    d=len(P)-1
    for i,a in enumerate(P):
        k = d-i
        coef = f"{a:.10g}"
        if k==0: terms.append(f"({coef})")
        elif k==1: terms.append(f"({coef})*n")
        else: terms.append(f"({coef})*pow(n,{k})")
    return "(" + "+".join(terms) + ")"

def piecewise_expr(n, y, deg):
    parts=[]
    for s in range(S):
        lo = indices[s]; hi = indices[s+1]-1 if s<S-1 else indices[s+1]-1
        expr = fit_poly_piece(n,y,lo,hi,deg)
        cond = f"between(n,{lo},{hi})"
        parts.append(f"if({cond},{expr},")
    tail = "(0)"  # shouldn't be used
    return "".join(parts) + tail + ")"*S

cx_expr = piecewise_expr(N, cx, 3)   # cubic pieces for pan
cy_expr = piecewise_expr(N, cy, 3)
z_expr  = "clip(" + piecewise_expr(N, z, 2) + ",1.0,1.55)"  # quadratic pieces for zoom

with open(out_ps1,"w",encoding="utf-8") as f:
    f.write("$cxExpr = '=%s'\n" % cx_expr)
    f.write("$cyExpr = '=%s'\n" % cy_expr)
    f.write("$zExpr  = '=%s'\n" % z_expr)
