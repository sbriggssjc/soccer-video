import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--in", required=True)
args = ap.parse_args()

source = getattr(args, "in")
cap = cv2.VideoCapture(source)
assert cap.isOpened()
fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("fps:", fps, "frames:", frames, "duration(s):", frames / (fps or 30.0))
