import sys, csv, ast, numpy as np, cv2

stable, ball_csv, vars_ps1, out_mp4 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

# read tuned center/zoom by recomputing (same as tuner wrote out)
# We'll parse width/height/x/y expressions directly from the ps1vars strings for consistency.

def read_vars(path):
    kv={}
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            if line.startswith("$"):
                k=line.split("=",1)[0].strip()
                v=line.split("=",1)[1].strip()
                kv[k]=v
    return kv

V=read_vars(vars_ps1)
cx_expr = V.get("$cxExpr","=in_w/2").lstrip("=").strip()
cy_expr = V.get("$cyExpr","=in_h/2").lstrip("=").strip()
z_expr  = V.get("$zExpr","=1").lstrip("=").strip()
safety  = float(V.get("$Safety","1.08"))

cap=cv2.VideoCapture(stable)
if not cap.isOpened(): raise SystemExit("cannot open stable video")
W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS=cap.get(cv2.CAP_PROP_FPS) or 60.0

# helper to eval ffmpeg-like expr with variables (n,in_w,in_h) safely
import math
def eval_expr(expr, n):
    # minimal safe namespace
    local={"n":float(n),"in_w":float(W),"in_h":float(H),
           "max":max,"min":min,"clip":lambda v,a,b: max(a,min(b,v)),
           "floor":math.floor,"ceil":math.ceil}
    # caret power -> python
    expr_py=expr.replace("^","**")
    return float(eval(expr_py,{"__builtins__":{}},local))

# derive per-frame box from tuned exprs
def box_for_n(n):
    cx = eval_expr(cx_expr,n)
    cy = eval_expr(cy_expr,n)
    z  = eval_expr(z_expr,n)
    w = math.floor( min(((H*9/16)/(z*safety)), W) /2 )*2
    h = math.floor( min((H/(z*safety)), H) /2 )*2
    x = math.floor( max(0, min(cx - w/2, W - w)) /2 )*2
    y = math.floor( max(0, min(cy - h/2, H - h)) /2 )*2
    return int(x),int(y),int(w),int(h),int(round(cx)),int(round(cy))

# writer
fourcc=cv2.VideoWriter_fourcc(*"avc1")
out=cv2.VideoWriter(out_mp4,fourcc, FPS, (W,H))
n=0
ok,frm=cap.read()
while ok:
    x,y,w,h,cx,cy = box_for_n(n)
    # yellow rectangle
    cv2.rectangle(frm, (x,y), (x+w-1,y+h-1), (0,255,255), 3)
    # red crosshair
    cv2.line(frm, (cx-22,cy), (cx+22,cy), (0,0,255), 3)
    cv2.line(frm, (cx,cy-22), (cx,cy+22), (0,0,255), 3)
    out.write(frm)
    n+=1
    ok,frm=cap.read()
cap.release(); out.release()
