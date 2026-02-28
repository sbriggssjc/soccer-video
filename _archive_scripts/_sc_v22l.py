import py_compile, sys
files = [
    r"D:\Projects\soccer-video\tools\ball_telemetry.py",
    r"D:\Projects\soccer-video\tools\render_follow_unified.py",
]
ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"OK: {f}", flush=True)
    except py_compile.PyCompileError as e:
        print(f"FAIL: {e}", flush=True)
        ok = False
if ok:
    print("ALL PASSED", flush=True)
else:
    print("ERRORS FOUND", flush=True)
    sys.exit(1)
