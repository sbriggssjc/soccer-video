"""Make tiny individual verification frames"""
import cv2, os

src = r'D:\Projects\soccer-video\_optflow_track\verify_frames'
dst = r'D:\Projects\soccer-video\out\verify'
os.makedirs(dst, exist_ok=True)

key = ['f0090.png','f0140.png','f0270.png','f0290.png','f0314.png','f0330.png','f0360.png','f0400.png','f0425.png','f0480.png']
for fn in key:
    path = os.path.join(src, fn)
    if not os.path.exists(path): continue
    img = cv2.imread(path)
    small = cv2.resize(img, (384, 216))
    out = os.path.join(dst, fn.replace('.png', '.jpg'))
    cv2.imwrite(out, small, [cv2.IMWRITE_JPEG_QUALITY, 50])
    sz = os.path.getsize(out)
    print(f"  {fn} -> {sz//1024}KB")
