@echo off
cd /d D:\Projects\soccer-video
python tools/render_follow_unified.py ^
  --in "out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4" ^
  --src "out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4" ^
  --out "out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__optflow_v2.mp4" ^
  --portrait 1080x1920 --preset cinematic --fps 24 --diagnostics --use-ball-telemetry
