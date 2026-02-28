import subprocess, os
out = r"D:\Projects\soccer-video\_debug_frames"
os.makedirs(out, exist_ok=True)
src = r"D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4"
reel = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__v22i_phantom_cleanup.mp4"
frames = [25,30,35,40,45,120,130,140,150,160,170,310,315,320,325,330,335,340,460,465,470]
for f in frames:
    subprocess.run(["ffmpeg","-y","-i",src,"-vf",f"select=eq(n\\,{f})","-vframes","1","-q:v","2",os.path.join(out,f"src_f{f:04d}.jpg")],capture_output=True)
    subprocess.run(["ffmpeg","-y","-i",reel,"-vf",f"select=eq(n\\,{f})","-vframes","1","-q:v","2",os.path.join(out,f"reel_f{f:04d}.jpg")],capture_output=True)
    print(f"Extracted frame {f}")
print("DONE")
