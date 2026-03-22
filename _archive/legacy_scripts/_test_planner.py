import sys, os, traceback
os.chdir(r"D:\Projects\soccer-video")
sys.path.insert(0, "tools")
try:
    import render_follow_unified as rfu
    print("Import OK")
    src = open("tools/render_follow_unified.py").read()
    for c in ["dz_edge_bypass","_near_edge","_kv_alpha_floor","_antic_frames","_CROSSOVER"]:
        print(f"  {c}: {'FOUND' if c in src else 'MISSING'}")
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
