import json, os, sys, cv2

def msg(s): print(s, flush=True)

def clamp(v,a,b): return max(a, min(b, v))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--anchors", type=int, default=5)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.inp)
    if not cap.isOpened():
        print(f"[ERR] Cannot open {args.inp}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    idx   = 0

    def grab(i):
        i = clamp(i, 0, total-1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        return (i, frame) if ok else (i, None)

    anchors = []
    win = "Select: s=box  J/K=±1  A/D=±30  W/S=±150  ENTER=save box  SPACE=next anchor  Q=quit"
    msg("Select a ROI and then press ENTER (repeat until you reach the requested number).")
    msg("Keys: J/K=1  A/D=30  W/S=150  S=draw box  ENTER=confirm box  SPACE=next anchor  Q=quit")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    i, frame = grab(idx)
    box = None
    while True:
        if frame is None: break
        disp = frame.copy()
        text = f"frame {i+1}/{total} | anchors {len(anchors)}/{args.anchors}"
        cv2.putText(disp, text, (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow(win, disp)
        k = cv2.waitKey(0) & 0xFF

        if k in (ord('q'), 27):
            break
        elif k == ord('j'):  # -1
            i, frame = grab(i-1)
            box=None
        elif k == ord('k'):  # +1
            i, frame = grab(i+1)
            box=None
        elif k == ord('a'):  # -30
            i, frame = grab(i-30)
            box=None
        elif k == ord('d'):  # +30
            i, frame = grab(i+30)
            box=None
        elif k == ord('w'):  # +150
            i, frame = grab(i+150)
            box=None
        elif k == ord('s'):  # -150
            i, frame = grab(i-150)
            box=None
        elif k == ord('S'):  # select box
            b = cv2.selectROI(win, frame, showCrosshair=True, fromCenter=False)
            if b is not None and all(v>0 for v in b):
                box = tuple(int(v) for v in b)
        elif k in (10,13):   # ENTER to confirm current box
            if box is not None:
                anchors.append({"frame": int(i), "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])]})
                msg(f"[OK] anchor {len(anchors)} @ frame {i} -> {box}")
                box=None
                if len(anchors) >= args.anchors:
                    break
        elif k == 32:  # SPACE -> next anchor prompt (without forcing box; not recommended)
            anchors.append({"frame": int(i), "bbox": None})
            msg(f"[NOTE] anchor {len(anchors)} @ frame {i} (no box)")
            if len(anchors) >= args.anchors:
                break

    cv2.destroyAllWindows()
    if len(anchors)==0:
        print("[CANCEL] no anchors selected"); sys.exit(2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"anchors": anchors, "total_frames": total, "fps": fps}, f, indent=2)
    print(f"[OK] Wrote {args.out} with {len(anchors)} anchor(s).")

if __name__ == "__main__":
    main()
