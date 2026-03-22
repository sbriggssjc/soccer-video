import os

base = "D:/Projects/soccer-video/out/portrait_reels"
games = [
    "2026-02-23__TSC_vs_NEOFC",
    "2026-02-23__TSC_vs_Greenwood",
    "2026-03-01__TSC_vs_North_OKC",
]

for game in games:
    d = os.path.join(base, game)
    files = [f for f in os.listdir(d) if not f.startswith(".")]
    finals = [f for f in files if "FINAL" in f]
    non_finals = [f for f in files if "FINAL" not in f]
    print(f"\n=== {game} ===")
    print(f"  Total: {len(files)}, FINAL: {len(finals)}, Other: {len(non_finals)}")
    if non_finals:
        # Group by type
        portraits = [f for f in non_finals if "portrait" in f and f.endswith(".mp4")]
        archives = [f for f in non_finals if f == "_archive" or os.path.isdir(os.path.join(d, f))]
        other = [f for f in non_finals if f not in portraits and f not in archives]
        if portraits:
            print(f"  Portrait intermediates: {len(portraits)}")
            for f in sorted(portraits)[:5]:
                sz = os.path.getsize(os.path.join(d, f)) / (1024*1024)
                print(f"    {f} ({sz:.1f} MB)")
            if len(portraits) > 5:
                print(f"    ... and {len(portraits)-5} more")
        if archives:
            print(f"  Directories: {archives}")
        if other:
            print(f"  Other files: {len(other)}")
            for f in sorted(other)[:10]:
                sz = os.path.getsize(os.path.join(d, f)) / (1024*1024) if os.path.isfile(os.path.join(d, f)) else 0
                print(f"    {f} ({sz:.1f} MB)")
            if len(other) > 10:
                print(f"    ... and {len(other)-10} more")

    # Total size
    total_sz = sum(os.path.getsize(os.path.join(d, f)) for f in files if os.path.isfile(os.path.join(d, f)))
    final_sz = sum(os.path.getsize(os.path.join(d, f)) for f in finals if os.path.isfile(os.path.join(d, f)))
    other_sz = total_sz - final_sz
    print(f"  Size - FINAL: {final_sz/(1024*1024):.0f} MB, Other: {other_sz/(1024*1024):.0f} MB")
