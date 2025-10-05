import sys, csv, numpy as np

in_csv, out_ps1 = sys.argv[1], sys.argv[2]

# Load
N=[]; cx=[]; cy=[]; conf=[]; w=h=fps=None
with open(in_csv,"r",newline="") as f:
    rd=csv.DictReader(f)
    for r in rd:
        N.append(int(r["n"]))
        cx.append(float(r["cx"])); cy.append(float(r["cy"])); conf.append(float(r["conf"]))
        w = int(r["w"]); h = int(r["h"]); fps = float(r["fps"])
N=np.array(N); cx=np.array(cx); cy=np.array(cy); conf=np.array(conf)
if len(N)==0: raise SystemExit("Empty CSV")

def ema(a, alpha):
    out=a.copy()
    for i in range(1,len(a)): out[i] = alpha*a[i] + (1-alpha)*out[i-1]
    return out

# Fill tiny gaps (confidence gate)
for i in range(1,len(cx)):
    if conf[i] < 0.12:
        cx[i]=cx[i-1]; cy[i]=cy[i-1]

# Base smooth
cx_s = ema(cx, 0.40); cy_s = ema(cy, 0.40)
cx_s = ema(cx_s, 0.22); cy_s = ema(cy_s, 0.22)

# Velocity/accel (per frame); speed for look-ahead + gain
vx, vy = np.gradient(cx_s), np.gradient(cy_s)
ax, ay = np.gradient(vx),  np.gradient(vy)
speed = np.sqrt(vx*vx + vy*vy)

# Look-ahead increases with speed (lead the action)
L = np.clip(8 + 0.30*speed, 8, 22)
cx_pred = np.clip(cx_s + vx*L + 0.5*ax*(L**2), 0, w-1)
cy_pred = np.clip(cy_s + vy*L + 0.5*ay*(L**2), 0, h-1)

# Variable gain: push harder when fast
speed_n = np.clip(speed/(0.30*max(w,h)), 0, 1)
k = 1.15 + 1.95*speed_n
camx = cx_s + k*(cx_pred - cx_s)
camy = cy_s + k*(cy_pred - cy_s)

# Fit compact polynomials (stable ffmpeg expr)
deg = 4 if len(N)>=60 else 3
Px = np.polyfit(N, camx, deg); Py = np.polyfit(N, camy, deg)
def poly_to_expr(P):
    d=len(P)-1; terms=[]
    for i,a in enumerate(P):
        p=d-i; c=f"{a:.10g}"
        terms.append(f"({c})" if p==0 else f"({c})*n^{p}" if p>1 else f"({c})*n")
    return "(" + "+".join(terms) + ")"
cx_expr = poly_to_expr(Px); cy_expr = poly_to_expr(Py)

# Zoom: wider by default + emergency near edges + panic when conf is low
margin_x, margin_y = 200.0, 240.0     # wider margins so ball + nearest player stay in
safety = 1.08
dx = np.minimum(camx, w-camx) - margin_x
dy = np.minimum(camy, h-camy) - margin_y
dx = np.clip(dx, 1, None); dy = np.clip(dy, 1, None)
z_need_x = (h*9/16)/(safety*2*dx)
z_need_y = (h)/(safety*2*dy)
z_needed = np.maximum(1.0, np.maximum(z_need_x, z_need_y))
z_base = np.minimum(1.30, np.maximum(1.0, 0.70*z_needed + 0.40))

# Emergency (edge proximity)
edge_frac = np.minimum.reduce([camx/(w+1e-6), (w-camx)/(w+1e-6), camy/(h+1e-6), (h-camy)/(h+1e-6)])
alpha = np.clip((0.28 - edge_frac)/0.28, 0, 1)  # faster near edges
z_emerg = np.maximum(z_base, z_needed)
z = (1-alpha)*z_base + alpha*z_emerg

# Panic zoom when confidence drops (quickly widen until reacquire)
conf_s = ema(conf, 0.35)
panic = np.clip((0.20 - conf_s)/0.20, 0, 1)     # 0 when conf>=0.20; →1 as conf→0
z = (1-0.6*panic)*z + (0.6*panic)*np.maximum(1.0, z_needed)

# Final smooth & small poly fit
z = ema(z, 0.24)
Pz = np.polyfit(N, z, min(3, len(N)-1))
z_expr = poly_to_expr(Pz)

with open(out_ps1,"w",encoding="utf-8") as f:
    f.write("$in_w = in_w\n$in_h = in_h\n")
    f.write(f"$cxExpr = '={cx_expr}'\n")
    f.write(f"$cyExpr = '={cy_expr}'\n")
    f.write(f"$zExpr  = '=clip({z_expr},1.0,1.30)'\n")
