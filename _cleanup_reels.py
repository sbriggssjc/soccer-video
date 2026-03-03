import os, shutil

base = "D:/Projects/soccer-video/out/portrait_reels"
games = [
    "2026-02-23__TSC_vs_NEOFC",
    "2026-02-23__TSC_vs_Greenwood",
]

total_freed = 0
for game in games:
    d = os.path.join(base, game)
    print(f"\n=== {game} ===")
    for item in sorted(os.listdir(d)):
        path = os.path.join(d, item)
        # Keep FINAL clips
        if "FINAL" in item:
            continue
        # Remove everything else
        if os.path.isdir(path):
            sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fns in os.walk(path) for f in fns)
            shutil.rmtree(path)
            print(f"  DEL dir:  {item} ({sz/(1024*1024):.1f} MB)")
            total_freed += sz
        elif os.path.isfile(path):
            sz = os.path.getsize(path)
            os.remove(path)
            print(f"  DEL file: {item} ({sz/(1024*1024):.1f} MB)")
            total_freed += sz

print(f"\nTotal freed: {total_freed/(1024*1024):.0f} MB")
print("Done")
