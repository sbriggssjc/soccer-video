"""Clean Feb 21 Greenwood intermediates from portrait reels folder."""
import os
from pathlib import Path

REEL_DIR = Path(r"D:\Projects\soccer-video\out\portrait_reels\2026-02-21__TSC_vs_Greenwood")

removed = 0
freed = 0
for f in sorted(REEL_DIR.iterdir()):
    if f.is_file() and "FINAL" not in f.name:
        size = f.stat().st_size
        try:
            os.remove(f)
            removed += 1
            freed += size
            print(f"  Removed {f.name} ({size/(1024*1024):.1f} MB)")
        except Exception as e:
            print(f"  FAILED {f.name}: {e}")

print(f"\nRemoved {removed} files, freed {freed/(1024*1024):.0f} MB")

# Also remove old FINAL files (they're 4K, we'll re-render at 1080p)
removed2 = 0
freed2 = 0
for f in sorted(REEL_DIR.glob("*FINAL*")):
    size = f.stat().st_size
    try:
        os.remove(f)
        removed2 += 1
        freed2 += size
        print(f"  Removed FINAL: {f.name} ({size/(1024*1024):.1f} MB)")
    except Exception as e:
        print(f"  FAILED {f.name}: {e}")

print(f"\nRemoved {removed2} old FINAL files, freed {freed2/(1024*1024):.0f} MB")
print(f"TOTAL: {removed + removed2} files, {(freed + freed2)/(1024*1024):.0f} MB freed")
