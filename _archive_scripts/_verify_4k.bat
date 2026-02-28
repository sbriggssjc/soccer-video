@echo off
cd /d "D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC"
echo === V19 Original ===
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration,bit_rate -of csv=p=0 002__stable_v19.mp4
echo === V19 4K Upscale ===
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration,bit_rate -of csv=p=0 002__stable_v19__4K.mp4
echo Done.
