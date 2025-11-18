import csv
import os
from pathlib import Path

ROOT = Path(r"C:\\Users\\scott\\soccer-video")
status_csv = ROOT / "out" / "reports" / "pipeline_status.csv"
catalog_csv = ROOT / "out" / "catalog" / "clips_catalog.csv"
catalog_csv.parent.mkdir(parents=True, exist_ok=True)

def find_final_portrait(clip_id: str) -> str:
    base = ROOT / "out" / "portrait_reels" / "clean"
    # Primary pattern
    candidate = base / f"{clip_id}_WIDE_portrait_FINAL.mp4"
    if candidate.exists():
        return str(candidate)
    # Fallback: any *_portrait_FINAL starting with ClipID
    if base.exists():
        for p in base.rglob(f"{clip_id}*_portrait_FINAL.mp4"):
            return str(p)
    return ""

def main():
    if not status_csv.exists():
        raise SystemExit(f"Status CSV not found: {status_csv}")

    with status_csv.open("r", newline="", encoding="utf-8") as f_in, \
         catalog_csv.open("w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = [
            "ClipID",
            "MatchKey",
            "AtomicPath",
            "FinalPortraitPath",
            "GameTag",
            "PlayerTags",
            "ActionTags",
            "Notes",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            clip_id = row.get("ClipID", "")
            match_key = row.get("MatchKey", "")
            atomic = row.get("AtomicPath", "")

            if not clip_id or not atomic:
                continue

            final_portrait = find_final_portrait(clip_id)

            writer.writerow({
                "ClipID": clip_id,
                "MatchKey": match_key,
                "AtomicPath": atomic,
                "FinalPortraitPath": final_portrait,
                "GameTag": match_key,   # default; you can tweak later
                "PlayerTags": "",
                "ActionTags": "",
                "Notes": "",
            })

    print(f"Wrote catalog: {catalog_csv}")

if __name__ == "__main__":
    main()
