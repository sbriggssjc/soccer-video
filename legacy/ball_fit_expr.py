import sys, csv, numpy as np

in_csv, out_ps1 = sys.argv[1], sys.argv[2]

# --- Load track ---
N=[]; cx=[]; cy=[]; conf=[]
w=h=fps=None
with open(in_csv,"r",newline="") as f:
    rd=csv.DictReader(f)
    for row in rd:
        n=int(row["n"]); N.append(n)
        cx.append(float(row["cx"])); cy.append(float(row["cy"]))
        conf.append(float(row["conf"]))
        w = int(row["w"]); h = int(row["h"]); fps = float(row["fps"])
N=np.array(N); cx=np.array(cx); cy=np.array(cy); conf=np.array(conf)
if len(N)==0: raise SystemExit("Empty CSV")

# --- Smooth base path ---
def ema(arr, a):
    out=np.copy(arr)
    for i in range(1,len(arr)): out[i]=a*arr[i]+(1-a)*out[i-1]
    return out
# hold tiny gaps
for i in range(1,len(cx)):
    if conf[i] < 0.12:
        cx[i] = cx[i-1]; cy[i] = cy[i-1]
cx_s = ema(cx, 0.38); cy_s = ema(cy, 0.38)
cx_s = ema(cx_s, 0.20); cy_s = ema(cy_s, 0.20)

# --- Predictive look-ahead (velocity + acceleration) ---
# finite differences in *frames*
vx = np.gradient(cx_s); vy = np.gradient(cy_s)
ax = np.gradient(vx);  ay = np.gradient(vy)
# lookahead in frames grows with speed (faster ball ? further look)
speed = np.sqrt(vx*vx + vy*vy)
L = np.clip(6 + 0.20*speed, 6, 18)  # ~0.25–0.75s @24fps
cx_pred = cx_s + vx*L + 0.5*ax*(L**2)
cy_pred = cy_s + vy*L + 0.5*ay*(L**2)
# clamp predictions
cx_pred = np.clip(cx_pred, 0, w-1)
cy_pred = np.clip(cy_pred, 0, h-1)

# --- Variable tracking gain (harder push when speed/error is high) ---
# we simulate a single pass proportional control feeding the *predicted* target
# map gain from 1.1 .. 2.8 by normalized speed
speed_n = np.clip(speed / (0.35*max(w,h)), 0, 1)   # normalize roughly by frame size
k = 1.1 + 1.7*speed_n
# one-step correction toward predicted target:
camx = cx_s + k*(cx_pred - cx_s)
camy = cy_s + k*(cy_pred - cy_s)

# --- Fit compact polynomials (stable ffmpeg expr) ---
deg = 4 if len(N)>=60 else 3
Px = np.polyfit(N, camx, deg)
Py = np.polyfit(N, camy, deg)
def poly_to_expr(P):
    d=len(P)-1; terms=[]
    for i,a in enumerate(P):
        p=d-i; coef=f"{a:.10g}"
        if p==0: terms.append(f"({coef})")
        elif p==1: terms.append(f"({coef})*n")
        else: terms.append(f"({coef})*n^{p}")
    return "(" + "+".join(terms) + ")"
cx_expr = poly_to_expr(Px)
cy_expr = poly_to_expr(Py)

# --- Zoom: wider by default + emergency bubble near edges ---
margin_x = 180.0; margin_y = 220.0      # wider margins to keep ball + nearest player
safety   = 1.08
dx = np.minimum(camx, w-camx) - margin_x
dy = np.minimum(camy, h-camy) - margin_y
dx = np.clip(dx, 1, None); dy = np.clip(dy, 1, None)
z_need_x = (h*9/16)/(safety*2*dx)
z_need_y = (h)/(safety*2*dy)
z_needed = np.maximum(1.0, np.maximum(z_need_x, z_need_y))

# base zoom: bias wide; cap at 1.35 to keep context
z_base = np.minimum(1.35, np.maximum(1.0, 0.75*z_needed + 0.35))

# emergency bubble: if ball too close to edge, quickly meet z_needed
edge_frac = np.minimum(np.minimum(camx, w-camx)/w, np.minimum(camy, h-camy)/h)  # 0 at edge
alpha = np.clip((0.25 - edge_frac)/0.25, 0, 1)  # ramp to 1 when inside ~25% border
z = (1-alpha)*z_base + alpha*np.maximum(z_base, z_needed)

# final smoothing + fit small poly
z = ema(z, 0.22)
Pz = np.polyfit(N, z, min(3, len(N)-1))
z_expr = poly_to_expr(Pz)

with open(out_ps1,"w",encoding="utf-8") as f:
    f.write("$in_w = in_w\n$in_h = in_h\n")
    f.write(f"$cxExpr = '={cx_expr}'\n")
    f.write(f"$cyExpr = '={cy_expr}'\n")
    f.write(f"$zExpr  = '=clip({z_expr},1.0,1.35)'\n")
