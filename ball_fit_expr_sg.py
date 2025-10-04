import sys, csv, numpy as np
from math import ceil
try:
    from scipy.signal import savgol_filter
    USE_SG = True
except Exception:
    USE_SG = False

if len(sys.argv)<3: raise SystemExit("usage: ball_fit_expr_sg.py <in_csv> <out_vars_ps1>")
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
if len(N)<3: raise SystemExit("Too few samples")

# fill small gaps by hold
for i in range(1,len(cx)):
    if conf[i] < 0.2:
        cx[i] = cx[i-1]; cy[i] = cy[i-1]

# basic EMA
def ema(arr, a):
    out=np.copy(arr)
    for i in range(1,len(arr)): out[i]=a*arr[i]+(1-a)*out[i-1]
    return out

cx1, cy1 = ema(cx, 0.35), ema(cy, 0.35)
cx2, cy2 = ema(cx1, 0.20), ema(cy1, 0.20)

# Savitzky–Golay (if available) to remove micro wiggles
if USE_SG:
    # choose odd window around ~0.25s, order 3
    win = max(5, int(round((fps*0.25)))|1)  # force odd
    cx2 = savgol_filter(cx2, win_length:=win, polyorder=3, mode='interp')
    cy2 = savgol_filter(cy2, win_length:=win, polyorder=3, mode='interp')

# lead ahead a bit (predictive centering)
lead = int(round(fps*0.25))  # ~0.25s
def lead_shift(a, k):
    if k<=0: return a
    pad = np.repeat(a[-1], k)
    return np.concatenate([a[k:], pad])
cx3 = lead_shift(cx2, lead); cy3 = lead_shift(cy2, lead)

# margins → needed zoom
margin_x, margin_y = 120.0, 150.0
dx = np.minimum(cx3, w-cx3) - margin_x
dy = np.minimum(cy3, h-cy3) - margin_y
dx = np.clip(dx, 1, None); dy = np.clip(dy, 1, None)
safety = 1.05
z_need_x = (h*9/16)/(safety*2*dx)
z_need_y = (h)/(safety*2*dy)
z_needed = np.maximum(1.0, np.maximum(z_need_x, z_need_y))
# bias wide a bit; soft-limit and smooth
z = np.minimum(1.55, np.maximum(1.0, 0.75*z_needed + 0.25))
z = ema(z, 0.25)

# compact polynomial fits (FFmpeg-safe pow(n,k))
def poly_to_expr(P):
    d=len(P)-1; terms=[]
    for i,a in enumerate(P):
        k=d-i; coef=f"{a:.10g}"
        if k==0: terms.append(f"({coef})")
        elif k==1: terms.append(f"({coef})*n")
        else: terms.append(f"({coef})*pow(n,{k})")
    return "(" + "+".join(terms) + ")"

deg_xy = 4 if len(N)>=60 else 3
Px = np.polyfit(N, cx3, deg_xy)
Py = np.polyfit(N, cy3, deg_xy)
Pz = np.polyfit(N, z, min(3, max(1,len(N)-1)))

cx_expr = poly_to_expr(Px)
cy_expr = poly_to_expr(Py)
z_expr  = poly_to_expr(Pz)

with open(out_ps1,"w",encoding="utf-8") as f:
    f.write("$cxExpr = '=%s'\n" % cx_expr)
    f.write("$cyExpr = '=%s'\n" % cy_expr)
    f.write("$zExpr  = '=clip(%s,1.0,1.55)'\n" % z_expr)
