with open(r"D:\Projects\soccer-video\_tmp\test_result.txt", "w") as f:
    f.write("Python is working\n")
    import glob, os
    pattern = os.path.join(r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC", "*__portrait.mp4")
    files = sorted(glob.glob(pattern))
    f.write(f"Pattern: {pattern}\n")
    f.write(f"Found: {len(files)} files\n")
    for fn in files[:5]:
        f.write(f"  {os.path.basename(fn)}\n")
