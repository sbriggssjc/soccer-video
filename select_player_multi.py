import argparse, json, pathlib, cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="out_json", required=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened(): raise SystemExit(f"Cannot open {args.src}")
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    win = "Multi-Select Player — use frame slider; press A to add selection; ENTER to save; ESC to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cur = 0; anchors = []  # list of {frame,x,y,w,h}

    def show_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            vis = frame.copy()
            for a in anchors:
                if abs(a["frame"]-idx) <= 1:
                    x,y,w,h = map(int, (a["x"],a["y"],a["w"],a["h"]))
                    cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imshow(win, vis)

    def on_trackbar(v):
        nonlocal cur
        cur = int(v)
        show_frame(cur)

    cv2.createTrackbar("frame", win, 0, max(0,N-1), on_trackbar)
    show_frame(0)

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            anchors = []
            break
        elif key in (ord('a'), ord('A')):
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
            ok, frame = cap.read()
            if not ok: continue
            r = cv2.selectROI("Draw player box, ENTER to accept", frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Draw player box, ENTER to accept")
            if r is not None and r[2] > 4 and r[3] > 8:
                anchors.append({"frame": int(cur), "x": float(r[0]), "y": float(r[1]), "w": float(r[2]), "h": float(r[3])})
                show_frame(cur)
        elif key in (10,13):  # ENTER — save
            if anchors:
                anchors = sorted(anchors, key=lambda a: a["frame"])
                out = {"fps": fps, "anchors": anchors}
                p = pathlib.Path(args.out_json); p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("w", encoding="utf-8") as f: json.dump(out, f, indent=2)
                print(f"[OK] Wrote {p} with {len(anchors)} anchor(s).")
            else:
                print("[WARN] No anchors made.")
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
