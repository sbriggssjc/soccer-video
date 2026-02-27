import py_compile
for f in [r"D:\Projects\soccer-video\tools\ball_telemetry.py",
          r"D:\Projects\soccer-video\tools\render_follow_unified.py"]:
    py_compile.compile(f, doraise=True)
    print(f"OK: {f}", flush=True)
print("ALL PASSED", flush=True)
