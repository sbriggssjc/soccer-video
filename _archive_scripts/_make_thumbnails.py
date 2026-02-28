"""Make smaller thumbnails of the verification frames"""
import cv2, os

src = r'D:\Projects\soccer-video\_optflow_track\verify_frames'
dst = r'D:\Projects\soccer-video\_optflow_track\verify_thumbs'
os.makedirs(dst, exist_ok=True)

for fn in sorted(os.listdir(src)):
    if not fn.endswith('.png'): continue
    img = cv2.imread(os.path.join(src, fn))
    # Resize to 640x360 (1/3 size)
    small = cv2.resize(img, (640, 360))
    cv2.imwrite(os.path.join(dst, fn), small, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
print(f"Created thumbnails in {dst}")
for fn in sorted(os.listdir(dst)):
    sz = os.path.getsize(os.path.join(dst, fn))
    print(f"  {fn}: {sz//1024}KB")
