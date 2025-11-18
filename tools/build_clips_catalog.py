import csv
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
status_csv = ROOT / "out" / "reports" / "pipeline_status.csv"
catalog_csv = ROOT / "out" / "catalog" / "clips_catalog.csv"
catalog_csv.parent.mkdir(parents=True, exist_ok=True)
PORTRAIT_ROOT = ROOT / "out" / "portrait_reels"


def probe_video(path: Path) -> Dict[str, str]:
    info: Dict[str, str] = {"DurationSeconds": "", "Width": "", "Height": ""}
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return info
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return info
    if "format" in data and "duration" in data["format"]:
        info["DurationSeconds"] = str(data["format"].get("duration", ""))
    streams = data.get("streams", [])
    if streams:
        stream = streams[0]
        info["Width"] = str(stream.get("width", ""))
        info["Height"] = str(stream.get("height", ""))
    return info


def clip_id_from_name(name: str) -> str:
    base = name
    if "_portrait_FINAL" in base:
        base = base.split("_portrait_FINAL", 1)[0]
    if "__" in base:
        base = base.split("__", 1)[0]
    return base


def find_portraits() -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    if not PORTRAIT_ROOT.exists():
        return mapping

    def sort_key(path: Path) -> tuple:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (-mtime, str(path))

    for mp4 in PORTRAIT_ROOT.rglob("*_portrait_FINAL*.mp4"):
        clip = clip_id_from_name(mp4.name)
        entries = mapping.setdefault(clip, [])
        if mp4 not in entries:
            entries.append(mp4)

    for clip, files in mapping.items():
        files.sort(key=sort_key)
        if len(files) > 1:
            logging.warning(
                "Multiple portrait reels remain for %s; keeping all (%s)",
                clip,
                "; ".join(str(p) for p in files),
            )
    return mapping


def load_status() -> Dict[str, Dict[str, str]]:
    data: Dict[str, Dict[str, str]] = {}
    if not status_csv.exists():
        return data
    with status_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clip_id = row.get("ClipID") or row.get("ID")
            if not clip_id:
                continue
            data[clip_id] = row
    return data


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    portraits = find_portraits()
    status = load_status()
    clip_ids = sorted(set(portraits) | set(status))
    if not clip_ids:
        raise SystemExit("No clips found; render reels before building the catalog")

    with catalog_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "ClipID",
            "MatchKey",
            "AtomicPath",
            "PortraitPaths",
            "DurationSeconds",
            "Width",
            "Height",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for clip_id in clip_ids:
            row = status.get(clip_id, {})
            atomic = row.get("AtomicPath") or ""
            match = row.get("MatchKey") or ""
            portrait_list = portraits.get(clip_id, [])
            portrait_str = ";".join(str(p) for p in portrait_list)
            metadata = (
                probe_video(portrait_list[0])
                if portrait_list
                else {"DurationSeconds": "", "Width": "", "Height": ""}
            )
            writer.writerow(
                {
                    "ClipID": clip_id,
                    "MatchKey": match,
                    "AtomicPath": atomic,
                    "PortraitPaths": portrait_str,
                    "DurationSeconds": metadata.get("DurationSeconds", ""),
                    "Width": metadata.get("Width", ""),
                    "Height": metadata.get("Height", ""),
                }
            )

    print(f"Wrote catalog: {catalog_csv}")


if __name__ == "__main__":
    main()
