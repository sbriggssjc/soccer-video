import subprocess, os
tiny = r"D:\Projects\soccer-video\_debug_tiny"
out = r"D:\Projects\soccer-video\_debug_composite"
os.makedirs(out, exist_ok=True)

# Zone 1: f25-45 (defender receives ball)
# Make a side-by-side composite: source on top, reel on bottom for key frames
zones = {
    "zone1_f25_f35_f45": [25, 35, 45],
    "zone3_f315_f325_f335": [315, 325, 335],
    "zone4_f460_f466_f470": [460, 466, 470],
}

for name, frames in zones.items():
    # Build ffmpeg filter to tile source and reel frames
    inputs = []
    for f in frames:
        src = os.path.join(tiny, f"src_f{f:04d}.png")
        reel = os.path.join(tiny, f"reel_f{f:04d}.png")
        if os.path.exists(src):
            inputs.append(src)
        if os.path.exists(reel):
            inputs.append(reel)
    
    if len(inputs) >= 2:
        cmd = ["ffmpeg", "-y"]
        for inp in inputs:
            cmd += ["-i", inp]
        n = len(inputs)
        # Stack: 3 columns (one per frame), 2 rows (source on top, reel on bottom)
        ncols = len(frames)
        cmd += ["-filter_complex", 
                f"{''.join(f'[{i}:v]' for i in range(0, n, 2))}hstack=inputs={ncols}[top];"
                f"{''.join(f'[{i}:v]' for i in range(1, n, 2))}hstack=inputs={ncols}[bot];"
                f"[top][bot]vstack=inputs=2[out]",
                "-map", "[out]", 
                os.path.join(out, f"{name}.png")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"ERROR {name}: {r.stderr[-200:]}")
        else:
            print(f"OK: {name}")

print("DONE", flush=True)
