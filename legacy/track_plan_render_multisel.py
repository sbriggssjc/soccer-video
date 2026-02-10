import argparse, json, math, pathlib, traceback, cv2, numpy as np

def clamp(v,a,b): return a if v<a else (b if v>b else v)

def hsv_hist(img, box):
    x,y,w,h = [int(round(t)) for t in box]
    x=max(0,x); y=max(0,y); w=max(1,w); h=max(1,h)
    roi = img[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1],None,[32,32],[0,180,0,256])
    cv2.normalize(hist,hist,0,1,cv2.NORM_MINMAX)
    return hist

def bhatta(a,b): return cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)

def iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ax2,ay2=ax+aw,ay+ah; bx2,by2=bx+bw,by+bh
    ix1,iy1=max(ax,bx),max(ay,by); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1); inter=iw*ih; u=aw*ah + bw*bh - inter
    return inter/u if u>0 else 0.0

def yolo_persons(frame, conf=0.18):
    try:
        from ultralytics import YOLO
        if not hasattr(yolo_persons, "model"):
            yolo_persons.model = YOLO("yolov8m.pt")
        pred = yolo_persons.model.predict(frame, classes=[0], conf=conf, iou=0.5, verbose=False)
        out=[]
        if pred and pred[0].boxes is not None:
            for (x1,y1,x2,y2) in pred[0].boxes.xyxy.cpu().numpy():
                out.append((float(x1),float(y1),float(x2-x1),float(y2-y1)))
        return out
    except Exception:
        return []  # YOLO optional  we can still follow on optical flow

def sample_points(box, grid=5):
    x,y,w,h=box
    pts=[]
    for i in range(grid):
        for j in range(grid+1):
            pts.append([x+(i+0.5)*w/grid, y+(j+0.5)*h/(grid+1)])
    return np.float32(pts).reshape(-1,1,2)

def bbox_from_points(pts):
    xs=pts[:,0,0]; ys=pts[:,0,1]
    x1=float(np.percentile(xs, 15)); x2=float(np.percentile(xs, 85))
    y1=float(np.percentile(ys, 15)); y2=float(np.percentile(ys, 85))
    return (x1,y1,max(8.0,x2-x1),max(16.0,y2-y1))

def choose_candidate(cands, prev_box, target_hist):
    if not cands: return None, 1.0
    px,py,pw,ph=prev_box; pcx, pcy=px+pw*0.5, py+ph*0.5
    best=None; best_cost=1e9
    for (box, hist, iouv) in cands:
        bx,by,bw,bh=box; bcx,bcy=bx+bw*0.5, by+bh*0.5
        app = bhatta(target_hist, hist)          # 0 is best
        di  = 1.0 - iouv
        dc  = math.hypot(bcx-pcx, bcy-pcy) / 80.0
        cost = 0.60*app + 0.25*di + 0.15*dc
        if cost < best_cost: best_cost, best = cost, box
    return best, best_cost

def polyfit_expr(ns, vs, deg):
    ns=np.asarray(ns,float); vs=np.asarray(vs,float)
    deg=min(deg, max(1,len(ns)-1))
    cs=np.polyfit(ns,vs,deg)
    terms=[]; p=deg
    for c in cs:
        if abs(c)<1e-12: p-=1; continue
        terms.append( f"({c:.10f})*n^{p}" if p>1 else (f"({c:.10f})*n" if p==1 else f"({c:.10f})") )
        p-=1
    return "(" + "+".join(terms) + ")" if terms else "(0)"

def moving_avg_future(x, L):
    n=len(x); y=np.empty(n)
    for i in range(n):
        j=min(n, i+L)
        y[i]=np.mean(x[i:j]) if j>i else x[i]
    return y

def vel_accel_limit(p, vmax, amax):
    p=np.array(p,float); v=np.zeros_like(p); out=np.zeros_like(p); out[0]=p[0]
    for i in range(1,len(p)):
        dv = p[i]-out[i-1]
        dv = max(-vmax, min(vmax, dv))
        dv = max(v[i-1]-amax, min(v[i-1]+amax, dv))
        v[i]=dv; out[i]=out[i-1]+v[i]
    return out

def zoom_for_subject(W,H,cx,cy,w,h,zmin,zmax,margin):
    half_h_need=(h/2.0)*(1.0+margin)
    half_w_need=(w/2.0)*(1.0+margin)
    z_req = min( H/(2.0*max(half_h_need,1.0)), (H*9/16)/(2.0*max(half_w_need,1.0)) )
    # keep edges in view
    z_edge_h = H/(2.0*max(24.0, min(cy, H-cy)))
    z_edge_w = (H*9/16)/(2.0*max(24.0, min(cx, W-cx)))
    z_low = max(zmin, z_edge_h, z_edge_w, 1.0)
    return float(max(z_low, min(zmax, z_req)))

def write_center_vars(vars_out, W, H):
    p=pathlib.Path(vars_out); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w",encoding="utf-8") as f:
        f.write(f"$cxExpr = \"= ({W/2.0:.3f})\"\n")
        f.write(f"$cyExpr = \"= ({H/2.0:.3f})\"\n")
        f.write(f"$zExpr  = \"= (1.00)\"\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",    dest="src", required=True)
    ap.add_argument("--sel",   dest="sel_json", required=True)  # anchors JSON
    ap.add_argument("--vars",  dest="vars_out", required=True)
    ap.add_argument("--debug", dest="debug_mp4", required=True)
    ap.add_argument("--out",   dest="out_mp4", required=True)
    # planner knobs (tuned for tighter, more responsive follow)
    ap.add_argument("--lookahead", type=int, default=10)
    ap.add_argument("--vmax", type=float, default=110.0)
    ap.add_argument("--amax", type=float, default=22.0)
    ap.add_argument("--edge_margin", type=float, default=0.30)
    ap.add_argument("--z_min", type=float, default=1.00)
    ap.add_argument("--z_max", type=float, default=1.12)
    ap.add_argument("--reacq_every", type=int, default=4)
    ap.add_argument("--bad_cost", type=float, default=0.36)
    args = ap.parse_args()

    errlog = pathlib.Path(args.vars_out).with_suffix(".error.log")
    try:
        sel = json.load(open(args.sel_json,"r",encoding="utf-8"))
        anchors = sel.get("anchors", [])
        if not anchors: raise RuntimeError("No anchors in selection JSON.")
        fps = sel.get("fps", 30.0)

        cap=cv2.VideoCapture(args.src)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open {args.src}")
        W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # anchor state
        a_idx=0
        cur_anchor = anchors[a_idx]
        next_anchor = anchors[a_idx+1] if a_idx+1<len(anchors) else None

        def set_anchor(frame_idx):
            nonlocal cur_anchor, a_idx, next_anchor, cur_box, target_hist, pts
            cur_anchor = anchors[a_idx]
            next_anchor = anchors[a_idx+1] if a_idx+1<len(anchors) else None
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_anchor["frame"])
            ok, fr = cap.read()
            if ok:
                cur_box = (cur_anchor["x"], cur_anchor["y"], cur_anchor["w"], cur_anchor["h"])
                target_hist = hsv_hist(fr, cur_box)
                pts = sample_points(cur_box)
            return ok

        cur_box=(0,0,32,64); target_hist=None; pts=None
        ok = set_anchor(anchors[0]["frame"])
        if not ok: raise RuntimeError("Failed to initialize at first anchor.")

        fourcc=cv2.VideoWriter_fourcc(*"mp4v")
        dbg = cv2.VideoWriter(args.debug_mp4, fourcc, fps, (W,H))

        lk=dict(winSize=(41,41), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        centers=[]; widths=[]; heights=[]; ns=[]
        prev_gray=None

        for n in range(N):
            ok, frame = cap.read()
            if not ok: break
            gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if next_anchor and n >= next_anchor["frame"]:
                a_idx += 1
                set_anchor(n)

            if prev_gray is None:
                prev_gray = gray.copy()
            new_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **lk)
            prop_box = cur_box
            if new_pts is not None and st is not None and int(st.sum()) >= 8:
                prop_box = bbox_from_points(new_pts)

            cands=[]
            phist = hsv_hist(frame, prop_box)
            cands.append( (prop_box, phist, iou(cur_box, prop_box)) )

            do_reacq = (n % args.reacq_every)==0
            est_app = bhatta(target_hist, phist)
            est_cost = 0.60*est_app + 0.25*(1.0 - iou(cur_box, prop_box))
            if est_cost > args.bad_cost: do_reacq = True

            if do_reacq:
                for d in yolo_persons(frame, conf=0.18):
                    cands.append( (d, hsv_hist(frame,d), iou(cur_box,d)) )

            best, _ = choose_candidate(cands, cur_box, target_hist)
            if best is None: best = prop_box

            cur_box = best
            pts = sample_points(cur_box)
            prev_gray=gray

            x,y,w,h = cur_box
            cx,cy = x+w*0.5, y+h*0.5
            centers.append((float(cx),float(cy)))
            widths.append(float(max(8.0,w))); heights.append(float(max(16.0,h)))
            ns.append(float(n))

            vis = frame.copy()
            cv2.rectangle(vis,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),2)
            cv2.circle(vis,(int(cx),int(cy)),4,(0,255,0),-1)
            dbg.write(vis)

        dbg.release(); cap.release()

        # SAFETY: always write vars (fallback to center if no samples)
        if not ns:
            write_center_vars(args.vars_out, W, H)
            print(f"[FALLBACK] wrote center vars -> {args.vars_out}")
            return

        cx = np.array([c[0] for c in centers]); cy = np.array([c[1] for c in centers])
        bw = np.array(widths); bh = np.array(heights)

        cx_f = moving_avg_future(cx,  args.lookahead)
        cy_f = moving_avg_future(cy,  args.lookahead)
        cx_s = vel_accel_limit(cx_f,  args.vmax, args.amax)
        cy_s = vel_accel_limit(cy_f,  args.vmax, args.amax)

        z_list=[]
        for i in range(len(ns)):
            sp = 0.0 if i==0 else min(1.0, ((cx_s[i]-cx_s[i-1])**2 + (cy_s[i]-cy_s[i-1])**2)**0.5/12.0)
            margin = args.edge_margin*(1.0 + 0.55*sp)
            z = zoom_for_subject(W,H, cx_s[i],cy_s[i], bw[i],bh[i], args.z_min, args.z_max, margin)
            if i>0: z = 0.25*z + 0.75*z_list[-1]
            z_list.append(z)

        cx_expr = polyfit_expr(ns, cx_s, 5)
        cy_expr = polyfit_expr(ns, cy_s, 5)
        z_expr  = polyfit_expr(ns, np.array(z_list), 3)

        p=pathlib.Path(args.vars_out); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w",encoding="utf-8") as f:
            f.write(f"$cxExpr = \"= {cx_expr}\"\n")
            f.write(f"$cyExpr = \"= {cy_expr}\"\n")
            f.write(f"$zExpr  = \"= {z_expr}\"\n")
        print(f"[OK] Vars -> {p}")

    except Exception as e:
        # ALWAYS write center vars + error log
        try:
            # Try to get W,H for a decent fallback
            cap=cv2.VideoCapture(args.src)
            W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1080
            H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1920
            cap.release()
        except:
            W,H=1080,1920
        write_center_vars(args.vars_out, W, H)
        msg="".join(traceback.format_exception(type(e), e, e.__traceback__))
        errlog.parent.mkdir(parents=True, exist_ok=True)
        errlog.write_text(msg, encoding="utf-8")
        print(f"[ERROR] wrote center vars and error log -> {errlog}")

if __name__ == "__main__":
    main()
