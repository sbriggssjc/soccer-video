@echo off
cd /d "D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC"
echo Starting 4K upscale of v19...
ffmpeg -y -i "002__stable_v19.mp4" -vf "scale=iw*2:ih*2:flags=lanczos,hqdn3d=2:1:2:3,unsharp=5:5:0.5:5:5:0.0" -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p "002__stable_v19__4K.mp4"
echo EXITCODE=%ERRORLEVEL%
echo Done.
