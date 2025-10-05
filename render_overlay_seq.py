import sys, csv, numpy as np, cv2, os, math, re

stable, vars_ps1, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)

def read_vars(path):
    kv={}
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            if line.startswith("$") and "=" in line:
                k,v=line.split("=",1)
                kv[k.strip()]=v.strip()
    return kv

def strip_quotes(s:str)->str:
    s=s.strip()
    if len(s)>=2 and ((s[0]=="'" and s[-1]=="'") or (s[0]=='"' and s[-1]=='"')):
        s=s[1:-1]
    return s

def normalize_ffmpeg_expr(raw:str)->str:
    """Make a PowerShell-written ffmpeg expr executable in Python:
       - strip quotes
       - remove literal .replace('==','=') tails
       - collapse any '==...' to a single leading '='
       - drop leading '=' (ffmpeg style)
       - map 'between('->'Between(', 'if('->'If('
       - map '^' -> '**'
    """
    s = strip_quotes(raw)
    # Remove any literal .replace('==','=') that we wrote in PS vars
    s = re.sub(r"\.replace\(\s*'=='\s*,\s*'='\s*\)\s*$","", s)
    # Collapse multiple leading '=' (sometimes we wrote '==expr')
    while s.startswith("=="):
        s = s[1:]
    if s.startswith("="):
        s = s[1:]
    # ffmpeg fn names & power
    s = s.replace("between(","Between(").replace("if(","If(")
    s = s.replace("^","**")
    return s

# safe helpers for eval
def Between(x,a,b): return 1.0 if (x>=a and x<=b) else 0.0
def If(cond,a,b):   return a if (cond!=0) else b
def clip(v,a,b):    return max(a,min(b,v))

V = read_vars(vars_ps1)
cx_raw = V.get("$cxExpr","=in_w/2")
cy_raw = V.get("$cyExpr","=in_h/2")
z_raw  = V.get("$zExpr","=1")
safety = float(V.get("$Safety","1.08"))

cx_expr = normalize_ffmpeg_expr(cx_raw)
cy_expr = normalize_ffmpeg_expr(cy_raw)
z_expr  = normalize_ffmpeg_expr(z_raw)

cap=cv2.VideoCapture(stable)
if not cap.isOpened(): raise SystemExit("cannot open stable video")
W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS=cap.get(cv2.CAP_PROP_FPS) or 60.0

def eval_expr(expr:str, n:int)->float:
    local = {"n": float(n), "in_w": float(W), "in_h": float(H),
             "Between":Between, "If":If, "clip":clip,
             "floor": math.floor, "ceil": math.ceil, "min": min, "max": max}
    val = eval(expr, {"__builtins__":{}}, local)
    # If any stray '=' snuck in, try once more after stripping
    if isinstance(val,str):
        val = float(eval(normalize_ffmpeg_expr(val), {"__builtins__":{}}, local))
    return float(val)

def box_for_n(n):
    cx = eval_expr(cx_expr,n); cy = eval_expr(cy_expr,n); z = eval_expr(z_expr,n)
    w = math.floor( min(((H*9/16)/(z*safety)), W) /2 )*2
    h = math.floor( min((H/(z*safety)), H) /2 )*2
    x = math.floor( max(0, min(cx - w/2, W - w)) /2 )*2
    y = math.floor( max(0, min(cy - h/2, H - h)) /2 )*2
    return int(x),int(y),int(w),int(h),int(round(cx)),int(round(cy))

idx=0
ok,frm=cap.read()
while ok:
    x,y,w,h,cx,cy = box_for_n(idx)
    cv2.rectangle(frm, (x,y), (x+w-1,y+h-1), (0,255,255), 3)  # yellow
    cv2.line(frm, (cx-22,cy), (cx+22,cy), (0,0,255), 3)       # red
    cv2.line(frm, (cx,cy-22), (cx,cy+22), (0,0,255), 3)
    cv2.imwrite(os.path.join(out_dir, f"{idx:06d}.png"), frm)
    idx+=1
    ok,frm=cap.read()
cap.release()

with open(os.path.join(out_dir,"fps.txt"),"w") as f: f.write(str(int(round(FPS))))
