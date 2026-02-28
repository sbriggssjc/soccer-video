import subprocess, os
out = r"D:\Projects\soccer-video\_debug_frames"
src = r"D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
reel = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__v22i_phantom_cleanup.mp4"
fps = 30.0
frames = [160,170,310,315,320,325,330,335,340,460,465,470]
for f in frames:
    t = f / fps
    for label, vid in [("src", src), ("reel", reel)]:
        outf = os.path.join(out, f"{label}_f{f:04d}.jpg")
        if os.path.exists(outf):
            continue
        subprocess.run(["ffmpeg","-y","-ss",f"{t:.3f}","-i",vid,"-vframes","1","-q:v","2",outf],capture_output=True)
    print(f"frame {f} done", flush=True)
print("DONE", flush=True)
