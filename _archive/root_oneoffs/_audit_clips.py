"""Audit where clips 005-009 final versions are stored."""
from pathlib import Path
import os

reels = Path("out/portrait_reels/2026-02-23__TSC_vs_NEOFC")
clean = Path("out/portrait_reels/clean")
desktop = Path(r"C:\Users\scott\Desktop")
telem = Path("out/telemetry")

for c in ["005", "006", "007", "008", "009"]:
    print(f"=== Clip {c} ===")
    
    # Reels dir
    for p in sorted(reels.glob(f"{c}__*")):
        sz = p.stat().st_size / (1024*1024)
        tag = "FINAL" if "FINAL" in p.name else ("portrait" if "portrait" in p.name else "other")
        print(f"  REELS [{tag:8s}]: {p.name} ({sz:.1f} MB)")
    
    # Clean dir
    for p in sorted(clean.glob(f"{c}__*NEOFC*")):
        sz = p.stat().st_size / (1024*1024)
        print(f"  CLEAN:          {p.name} ({sz:.1f} MB)")
    
    # Desktop
    for pattern in [f"clip_{c}*", f"*{c}*filmstrip*", f"review_{c}*"]:
        for p in sorted(desktop.glob(pattern)):
            sz = p.stat().st_size / (1024*1024)
            print(f"  DESKTOP:        {p.name} ({sz:.1f} MB)")
    
    # Telemetry backups
    baks = list(telem.glob(f"{c}__*2026-02-23*.bak")) + list(telem.glob(f"{c}__*2026-02-23*.1280bak"))
    if baks:
        print(f"  TELEMETRY:      {len(baks)} backup files")
    
    print()
