@echo off
cd /d D:\Projects\soccer-video
C:\Users\scott\AppData\Local\Programs\Python\Python313\python.exe -u tools\render_follow_unified.py --preset cinematic --in out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4 --out out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__polished_v12.mp4 --portrait 1080x1920 --no-draw-ball --diagnostics > _v12_log.txt 2>&1
echo EXITCODE=%ERRORLEVEL% >> _v12_log.txt
