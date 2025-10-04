import sys, csv, numpy as np
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
if len(N)==0: raise SystemExit("Empty CSV")

def ema(arr, alpha):
    out=np.copy(arr)
    for i in range(1,len(arr)):
        out[i]=alpha*arr[i]+(1-alpha)*out[i-1]
    return out

# fill small gaps by hold
for i in range(1,len(cx)):
    if conf[i] < 0.15:
        cx[i] = cx[i-1]
        cy[i] = cy[i-1]

# smooth
cx_s = ema(cx, 0.35); cy_s = ema(cy, 0.35)
cx_s = ema(cx_s, 0.18); cy_s = ema(cy_s, 0.18)

# poly fits
deg = 4 if len(N)>=60 else 3
Px = np.polyfit(N, cx_s, deg)
Py = np.polyfit(N, cy_s, deg)

def poly_to_expr(P):
    d=len(P)-1
    terms=[]
    for i,a in enumerate(P):
        powr=d-i
        coef=f"{a:.10g}"
        if powr==0: terms.append(f"({coef})")
        elif powr==1: terms.append(f"({coef})*n")
        else: terms.append(f"({coef})*n^{powr}")
    return "(" + "+".join(terms) + ")"

cx_expr = poly_to_expr(Px)
cy_expr = poly_to_expr(Py)

# zoom from safe margins
margin_x = 120.0; margin_y = 150.0
dx = np.minimum(cx_s, w-cx_s) - margin_x
dy = np.minimum(cy_s, h-cy_s) - margin_y
dx = np.clip(dx, 1, None); dy = np.clip(dy, 1, None)

safety = 1.05
z_need_x = (h*9/16)/(safety*2*dx)
z_need_y = (h)/(safety*2*dy)
z_needed = np.maximum(1.0, np.maximum(z_need_x, z_need_y))

z = np.minimum(1.55, np.maximum(1.0, 0.8*z_needed + 0.2))
z = ema(z, 0.20)
Pz = np.polyfit(N, z, min(3, len(N)-1))
z_expr = poly_to_expr(Pz)

with open(out_ps1,"w",encoding="utf-8") as f:
    f.write("$in_w = in_w\n$in_h = in_h\n")
    f.write(f"$cxExpr = '={cx_expr}'\n")
    f.write(f"$cyExpr = '={cy_expr}'\n")
    f.write(f"$zExpr  = '=clip({z_expr},1.0,1.55)'\n")
