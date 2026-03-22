@echo off
cd /d D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC

echo === PASS 1: Detect shake ===
ffmpeg -y -i 002__smooth_v18.mp4 -vf vidstabdetect=shakiness=5:accuracy=15:result=v18_transforms.trf -f null - 2>&1 | findstr /i "frame vidstab"

echo === PASS 2: Apply stabilization ===
ffmpeg -y -i 002__smooth_v18.mp4 -vf vidstabtransform=input=v18_transforms.trf:smoothing=15:interpol=bicubic:crop=black:zoom=3 -c:v libx264 -crf 17 -preset medium -pix_fmt yuv420p 002__stable_v19.mp4

echo EXITCODE=%ERRORLEVEL%