import cv2, os

d = "D:/Projects/soccer-video/out/portrait_reels/2026-02-23__TSC_vs_NEOFC"
clips = [f for f in sorted(os.listdir(d)) if "FINAL" in f]

for c in clips:
    cap = cv2.VideoCapture(os.path.join(d, c))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = frames / fps if fps > 0 else 0
    cap.release()
    print(f"{c[:3]}: fps={fps:.2f}  frames={frames}  dur={dur:.1f}s")
