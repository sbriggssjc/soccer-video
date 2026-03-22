@echo off
cd /d "D:\Projects\soccer-video\_tmp"
set SRC=D:/Projects/soccer-video/out/portrait_reels/2026-02-23__TSC_vs_NEOFC/009__2026-02-23__TSC_vs_NEOFC__CORNER__t850.00-t856.00__portrait.mp4
echo Testing vidstab on shortest clip (009, 6s)...
echo Pass 1: detect
ffmpeg -y -i "%SRC%" -vf "vidstabdetect=shakiness=5:accuracy=15:result=D:/Projects/soccer-video/_tmp/test.trf" -f null - 2>&1 | findstr /i "frame"
echo Pass 2: transform
ffmpeg -y -i "%SRC%" -vf "vidstabtransform=input=D:/Projects/soccer-video/_tmp/test.trf:smoothing=15:interpol=bicubic:crop=black:zoom=3" -c:v libx264 -crf 17 -preset medium -pix_fmt yuv420p D:/Projects/soccer-video/_tmp/test_stab.mp4 2>&1 | findstr /i "frame error"
echo EXITCODE=%ERRORLEVEL%
ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x D:/Projects/soccer-video/_tmp/test_stab.mp4
echo Done.
