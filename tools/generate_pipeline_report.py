#!/usr/bin/env python3
#!/usr/bin/env python3
"""Generate pipeline status report and PowerShell command script."""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

RE_CLIP_ID = re.compile(r"(?P<id>\d{3}__[^\\/]+__t[-\d.]+-t[-\d.]+)")
RE_MP4 = re.compile(r"(?P<path>[^\s\",']+\.mp4)", re.IGNORECASE)

STAGE_NAMES = [
    "Stage_Stabilized",
    "Stage_Follow",
    "Stage_Upscaled",
    "Stage_Enhanced",
    "Stage_Branded",
]

PRESET_DEFAULT = "cinematic"
PORTRAIT_DEFAULT = "1080x1920"
UPSCALE_DEFAULT = 2

@dataclass
class ClipStageInfo:
    match_key: str
    clip_id: str
    clip_path: Path
    selected_manual: bool
    stabilized_paths: List[Path]
    follow_paths: List[Path]
    upscaled_paths: List[Path]
    enhanced_paths: List[Path]
    branded_paths: List[Path]

    def best_source_for_follow(self) -> Path:
        if self.follow_paths:
            return self.follow_paths[0]
        if self.stabilized_paths:
            return self.stabilized_paths[0]
        return self.clip_path

    def best_source_for_upscale(self) -> Path:
        if self.follow_paths:
            return self.follow_paths[0]
        if self.stabilized_paths:
            return self.stabilized_paths[0]
        return self.clip_path

    def best_source_for_enhance(self) -> Path:
        if self.upscaled_paths:
            return self.upscaled_paths[0]
        if self.follow_paths:
            return self.follow_paths[0]
        if self.stabilized_paths:
            return self.stabilized_paths[0]
        return self.clip_path

    def best_source_for_brand(self) -> Path:
        if self.branded_paths:
            return self.branded_paths[0]
        if self.enhanced_paths:
            return self.enhanced_paths[0]
        if self.upscaled_paths:
            return self.upscaled_paths[0]
        if self.follow_paths:
            return self.follow_paths[0]
        if self.stabilized_paths:
            return self.stabilized_paths[0]
        return self.clip_path


def _win_path(path: Path) -> str:
    return str(path).replace("/", "\\")


def _collect_manual_selection(repo_root: Path) -> Dict[str, bool]:
    candidates = [
        repo_root / "events_selected.csv",
        repo_root / "events_selected_resolved.csv",
        repo_root / "events_selected_resolved.paths.csv",
        repo_root / "out" / "events_selected.csv",
    ]
    selected: Dict[str, bool] = {}

    def record_clip_id(clip_id: str) -> None:
        if clip_id:
            selected[clip_id] = True

    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                for row in reader:
                    for value in row:
                        if not value:
                            continue
                        for match in RE_MP4.finditer(value):
                            candidate_path = Path(match.group("path")).name
                            record_clip_id(Path(candidate_path).stem)
                        for match in RE_CLIP_ID.finditer(value):
                            record_clip_id(match.group("id"))
        except Exception:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for match in RE_MP4.finditer(text):
                record_clip_id(Path(match.group("path")).stem)
            for match in RE_CLIP_ID.finditer(text):
                record_clip_id(match.group("id"))
    return selected


def _glob_matching(root: Path, pattern: str) -> List[Path]:
    return [p for p in root.glob(pattern) if p.exists()]


def _find_stabilized(repo_root: Path, clip_stem: str) -> List[Path]:
    cine_root = repo_root / "out" / "autoframe_work" / "cinematic"
    results: List[Path] = []
    if not cine_root.exists():
        return results
    for folder in sorted(cine_root.glob(f"{clip_stem}*")):
        candidate = folder / "follow" / "stabilized.mp4"
        if candidate.exists():
            results.append(candidate)
    return results


def _find_follow_outputs(clip_path: Path) -> List[Path]:
    """Return any follow render variants associated with ``clip_path``.

    Historically the follow stage has produced multiple stabilized renders
    such as ``__CINEMATIC`` and ``__REALZOOM``. Recent follow automation also
    emits ``__FOLLOW.mp4`` for the direct camera track, so the discovery list
    explicitly includes that suffix alongside the existing variants.
    """

    parent = clip_path.parent
    stem = clip_path.stem
    suffixes = ["__CINEMATIC", "__GENTLE", "__REALZOOM", "__FOLLOW"]
    results: List[Path] = []
    for suffix in suffixes:
        candidate = parent / f"{stem}{suffix}.mp4"
        if candidate.exists():
            results.append(candidate)
    return results


def _find_upscaled(repo_root: Path, clip_stem: str) -> List[Path]:
    up_root = repo_root / "out" / "upscaled"
    results: List[Path] = []
    if not up_root.exists():
        return results
    for candidate in sorted(up_root.glob(f"{clip_stem}__x*.mp4")):
        if candidate.exists():
            results.append(candidate)
    return results


def _find_enhanced(paths: Sequence[Path]) -> List[Path]:
    results: List[Path] = []
    for src in paths:
        candidate = src.with_name(f"{src.stem}_ENH.mp4")
        if candidate.exists():
            results.append(candidate)
    return results


def _find_branded(repo_root: Path, clip_stem: str) -> List[Path]:
    candidates: List[Path] = []
    portrait_root = repo_root / "out" / "portrait_reels"
    if portrait_root.exists():
        for sub in ["clean", "branded", "postable"]:
            subdir = portrait_root / sub
            if not subdir.exists():
                continue
            for pattern in (
                f"{clip_stem}_portrait_FINAL*.mp4",
                f"{clip_stem}_portrait_BRAND*.mp4",
                f"{clip_stem}_portrait_POST*.mp4",
            ):
                for item in subdir.glob(pattern):
                    if item.exists():
                        candidates.append(item)
    # fallback: same folder as follow outputs
    alt = (repo_root / "out").glob(f"**/{clip_stem}_portrait_FINAL*.mp4")
    for item in alt:
        if item.exists() and item not in candidates:
            candidates.append(item)
    return candidates


def collect_clips(repo_root: Path) -> List[ClipStageInfo]:
    atomic_root = repo_root / "out" / "atomic_clips"
    selected_lookup = _collect_manual_selection(repo_root)
    clips: List[ClipStageInfo] = []

    if not atomic_root.exists():
        return clips

    for match_dir in sorted(p for p in atomic_root.iterdir() if p.is_dir()):
        match_key = match_dir.name
        for clip_path in sorted(match_dir.glob("*.mp4")):
            clip_id = clip_path.stem
            stabilized = _find_stabilized(repo_root, clip_id)
            follow = _find_follow_outputs(clip_path)
            upscaled = _find_upscaled(repo_root, clip_id)
            enhanced_sources = list({*follow, *upscaled, clip_path})
            enhanced = _find_enhanced(enhanced_sources)
            branded = _find_branded(repo_root, clip_id)
            clips.append(
                ClipStageInfo(
                    match_key=match_key,
                    clip_id=clip_id,
                    clip_path=clip_path,
                    selected_manual=selected_lookup.get(clip_id, False),
                    stabilized_paths=stabilized,
                    follow_paths=follow,
                    upscaled_paths=upscaled,
                    enhanced_paths=enhanced,
                    branded_paths=branded,
                )
            )
    return clips


def stage_done(paths: Sequence[Path]) -> bool:
    return any(path.exists() for path in paths)


def build_command_entry(stage: str, info: ClipStageInfo, repo_root: Path) -> Optional[Dict[str, object]]:
    atomic_path = info.clip_path
    clip_stem = info.clip_id

    if stage == "Stage_Stabilized":
        out_path = repo_root / "out" / "autoframe_work" / "cinematic" / clip_stem / "follow" / "stabilized.mp4"
        command = [
            "python",
            str(repo_root / "tools" / "render_follow_unified.py"),
            "--in",
            _win_path(atomic_path),
            "--preset",
            PRESET_DEFAULT,
            "--out",
            _win_path(out_path),
            "--clean-temp",
        ]
        input_path = atomic_path
    elif stage == "Stage_Follow":
        out_path = atomic_path.with_name(f"{clip_stem}.__CINEMATIC.mp4")
        source = info.best_source_for_follow()
        command = [
            "python",
            str(repo_root / "tools" / "render_follow_unified.py"),
            "--in",
            _win_path(source),
            "--preset",
            PRESET_DEFAULT,
            "--out",
            _win_path(out_path),
            "--clean-temp",
        ]
        input_path = source
    elif stage == "Stage_Upscaled":
        source = info.best_source_for_upscale()
        out_path = repo_root / "out" / "upscaled" / f"{clip_stem}__x{UPSCALE_DEFAULT}.mp4"
        upscale_py = repo_root / "tools" / "upscale.py"
        if upscale_py.exists():
            command = [
                "python",
                "-c",
                (
                    "import sys; sys.path.insert(0, r'{}'); "
                    "from upscale import upscale_video; "
                    "upscale_video(r'{}', scale={})"
                ).format(
                    _win_path(upscale_py.parent),
                    _win_path(source),
                    UPSCALE_DEFAULT,
                ),
            ]
        else:
            command = [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i",
                _win_path(source),
                "-vf",
                "scale=1080:1920:flags=lanczos",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "18",
                _win_path(out_path),
            ]
        input_path = source
    elif stage == "Stage_Enhanced":
        source = info.best_source_for_enhance()
        out_path = source.with_name(f"{Path(source).stem}_ENH.mp4")
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(repo_root / "tools" / "auto_enhance" / "auto_enhance.ps1"),
            "-In",
            _win_path(source),
        ]
        input_path = source
    elif stage == "Stage_Branded":
        source = info.best_source_for_brand()
        out_root = repo_root / "out" / "portrait_reels" / "clean"
        out_path = out_root / f"{clip_stem}_portrait_FINAL.mp4"
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(repo_root / "tools" / "tsc_brand.ps1"),
            "-In",
            _win_path(source),
            "-Out",
            _win_path(out_path),
            "-Aspect",
            "9x16",
        ]
        input_path = source
    else:
        return None

    return {
        "Stage": stage,
        "MatchKey": info.match_key,
        "ClipID": info.clip_id,
        "Input": _win_path(Path(input_path)),
        "Output": _win_path(Path(out_path)),
        "Command": command,
    }


def generate_reports(repo_root: Path) -> Tuple[List[ClipStageInfo], List[Dict[str, object]]]:
    clips = collect_clips(repo_root)
    commands: List[Dict[str, object]] = []

    for info in clips:
        stage_presence = {
            "Stage_Stabilized": stage_done(info.stabilized_paths),
            "Stage_Follow": stage_done(info.follow_paths),
            "Stage_Upscaled": stage_done(info.upscaled_paths),
            "Stage_Enhanced": stage_done(info.enhanced_paths),
            "Stage_Branded": stage_done(info.branded_paths),
        }
        for stage, done in stage_presence.items():
            if not done:
                entry = build_command_entry(stage, info, repo_root)
                if entry is not None:
                    commands.append(entry)

    return clips, commands


def write_status_csv(repo_root: Path, clips: Sequence[ClipStageInfo]) -> Path:
    out_dir = repo_root / "out" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pipeline_status.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "MatchKey",
                "ClipID",
                "AtomicPath",
                "SelectedManual",
                "Stage_Stabilized",
                "Stage_Follow",
                "Stage_Upscaled",
                "Stage_Enhanced",
                "Stage_Branded",
            ]
        )
        for info in clips:
            writer.writerow(
                [
                    info.match_key,
                    info.clip_id,
                    _win_path(info.clip_path),
                    "True" if info.selected_manual else "False",
                    "True" if stage_done(info.stabilized_paths) else "False",
                    "True" if stage_done(info.follow_paths) else "False",
                    "True" if stage_done(info.upscaled_paths) else "False",
                    "True" if stage_done(info.enhanced_paths) else "False",
                    "True" if stage_done(info.branded_paths) else "False",
                ]
            )
    return csv_path


def _ps_quote(item: str) -> str:
    return item.replace("'", "''")


def write_command_script(repo_root: Path, commands: Sequence[Dict[str, object]]) -> Path:
    out_dir = repo_root / "out" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / "pipeline_commands_to_run.ps1"

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    with script_path.open("w", encoding="utf-8") as handle:
        handle.write("param([bool]$WhatIf = $true)\n")
        handle.write("$ErrorActionPreference = 'Stop'\n")
        repo_root_str = str(repo_root).replace("\\", "\\\\")
        handle.write("$script:RepoRoot = '{}'\n".format(repo_root_str))
        handle.write(
            "$logPath = Join-Path $script:RepoRoot '{}'\n".format(
                "out\\logs\\advance_pipeline.{0}.log".format(timestamp)
            )
        )
        handle.write("$null = New-Item -ItemType Directory -Force -Path (Split-Path $logPath)\n")
        handle.write("$logStream = New-Item -ItemType File -Force -Path $logPath\n")
        handle.write("$commands = @()\n")
        for entry in commands:
            command_parts = entry["Command"]
            ps_array = ",".join("'" + _ps_quote(str(part)) + "'" for part in command_parts)
            handle.write(
                "${{commands}} += [pscustomobject]@{{ Stage='{}'; MatchKey='{}'; ClipID='{}'; Input='{}'; Output='{}'; Command=@({}) }}\n".format(
                    _ps_quote(str(entry["Stage"])),
                    _ps_quote(str(entry["MatchKey"])),
                    _ps_quote(str(entry["ClipID"])),
                    _ps_quote(str(entry["Input"])),
                    _ps_quote(str(entry["Output"])),
                    ps_array,
                )
            )
        handle.write("if ($commands.Count -eq 0) { Write-Host 'No pending stages.'; return }\n")
        handle.write(
            "foreach ($stageGroup in ($commands | Sort-Object Stage, MatchKey, ClipID | Group-Object Stage)) {\n"
        )
        handle.write("  Write-Host ('### Stage: {0}' -f $stageGroup.Name) -ForegroundColor Cyan\n")
        handle.write("  foreach ($matchGroup in ($stageGroup.Group | Group-Object MatchKey)) {\n")
        handle.write("    Write-Host ('# Match: {0}' -f $matchGroup.Name) -ForegroundColor Yellow\n")
        handle.write("    foreach ($entry in $matchGroup.Group) {\n")
        handle.write("      $inPath = $entry.Input\n")
        handle.write("      $outPath = $entry.Output\n")
        handle.write("      $cmd = $entry.Command\n")
        handle.write("      $cmdString = ($cmd | ForEach-Object { '\"' + $_ + '\"' }) -join ' ' \n")
        handle.write("      Add-Content -Path $logPath -Value $cmdString\n")
        handle.write("      $outDir = Split-Path -Parent $outPath\n")
        handle.write("      if ($outDir -and -not (Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }\n")
        handle.write("      $shouldSkip = $false\n")
        handle.write("      if ((Test-Path $inPath) -and (Test-Path $outPath)) {\n")
        handle.write("        $inItem = Get-Item $inPath\n")
        handle.write("        $outItem = Get-Item $outPath\n")
        handle.write("        if ($outItem.LastWriteTimeUtc -ge $inItem.LastWriteTimeUtc) { $shouldSkip = $true }\n")
        handle.write("      }\n")
        handle.write("      if ($shouldSkip) {\n")
        handle.write("        Write-Host ('[SKIP] {0} :: up-to-date' -f $cmdString)\n")
        handle.write("        Add-Content -Path $logPath -Value '[SKIP] up-to-date'\n")
        handle.write("        continue\n")
        handle.write("      }\n")
        handle.write("      Write-Host ('[RUN ] {0}' -f $cmdString)\n")
        handle.write("      if ($WhatIf) {\n")
        handle.write("        Write-Host '       WhatIf: not executed.' -ForegroundColor Green\n")
        handle.write("        Add-Content -Path $logPath -Value '[WhatIf] not executed'\n")
        handle.write("        continue\n")
        handle.write("      }\n")
        handle.write("      $psi = New-Object System.Diagnostics.ProcessStartInfo\n")
        handle.write("      $psi.FileName = $cmd[0]\n")
        handle.write("      if ($cmd.Count -gt 1) { $psi.Arguments = [string]::Join(' ', ($cmd[1..($cmd.Count-1)] | ForEach-Object { '\"' + $_ + '\"' })) }\n")
        handle.write("      $psi.WorkingDirectory = $script:RepoRoot\n")
        handle.write("      $psi.UseShellExecute = $false\n")
        handle.write("      $psi.RedirectStandardOutput = $true\n")
        handle.write("      $psi.RedirectStandardError = $true\n")
        handle.write("      $proc = [System.Diagnostics.Process]::Start($psi)\n")
        handle.write("      $stdout = $proc.StandardOutput.ReadToEnd()\n")
        handle.write("      $stderr = $proc.StandardError.ReadToEnd()\n")
        handle.write("      $proc.WaitForExit()\n")
        handle.write("      if ($stdout) { Add-Content -Path $logPath -Value $stdout }\n")
        handle.write("      if ($stderr) { Add-Content -Path $logPath -Value $stderr }\n")
        handle.write("      if ($proc.ExitCode -ne 0) {\n")
        handle.write("        Write-Warning ('Command exited with code {0}' -f $proc.ExitCode)\n")
        handle.write("        Add-Content -Path $logPath -Value ('ExitCode={0}' -f $proc.ExitCode)\n")
        handle.write("      }\n")
        handle.write("    }\n")
        handle.write("  }\n")
        handle.write("}\n")
        handle.write("Write-Host ('Log: {0}' -f $logPath)\n")
    return script_path


def print_stage_command_map() -> None:
    mapping = {
        "Stage_Stabilized": "python tools\\render_follow_unified.py --in <clip> --preset cinematic --out <out\\autoframe_work\\cinematic\\<ClipID>\\follow\\stabilized.mp4> --clean-temp",
        "Stage_Follow": "python tools\\render_follow_unified.py --in <source> --preset cinematic --out <clip.__CINEMATIC.mp4> --clean-temp",
        "Stage_Upscaled": "python -c \"from tools.upscale import upscale_video; upscale_video(r'<source>', scale=2)\"",
        "Stage_Enhanced": "powershell -File tools\\auto_enhance\\auto_enhance.ps1 -In <source>",
        "Stage_Branded": "powershell -File tools\\tsc_brand.ps1 -In <source> -Out <out\\portrait_reels\\clean\\<ClipID>_portrait_FINAL.mp4> -Aspect 9x16",
    }
    print("Stage → command map:")
    for stage in STAGE_NAMES:
        print(f"  {stage}: {mapping.get(stage, 'N/A')}")


def print_summary(clips: Sequence[ClipStageInfo]) -> None:
    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for info in clips:
        stats = summary[info.match_key]
        stats["Total"] += 1
        if info.selected_manual:
            stats["Selected"] += 1
        stats["Stage_Stabilized_done"] += int(stage_done(info.stabilized_paths))
        stats["Stage_Follow_done"] += int(stage_done(info.follow_paths))
        stats["Stage_Upscaled_done"] += int(stage_done(info.upscaled_paths))
        stats["Stage_Enhanced_done"] += int(stage_done(info.enhanced_paths))
        stats["Stage_Branded_done"] += int(stage_done(info.branded_paths))
    if not summary:
        print("No atomic clips found under out/atomic_clips.")
        return
    headers = [
        "MatchKey",
        "Total",
        "Selected",
        "Stab_Done",
        "Follow_Done",
        "Upscaled_Done",
        "Enhanced_Done",
        "Branded_Done",
    ]
    print("\nSummary by match:")
    print(",".join(headers))
    for match_key in sorted(summary.keys()):
        stats = summary[match_key]
        row = [
            match_key,
            str(stats.get("Total", 0)),
            str(stats.get("Selected", 0)),
            str(stats.get("Stage_Stabilized_done", 0)),
            str(stats.get("Stage_Follow_done", 0)),
            str(stats.get("Stage_Upscaled_done", 0)),
            str(stats.get("Stage_Enhanced_done", 0)),
            str(stats.get("Stage_Branded_done", 0)),
        ]
        print(",".join(row))


def find_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(8):
        if (cur / ".git").exists() or (cur / "out" / "atomic_clips").exists():
            return cur
        cur = cur.parent
    return start  # fallback


def main() -> None:
    here = Path(__file__).resolve()
    repo_root = find_repo_root(here.parent)
    clips, commands = generate_reports(repo_root)
    status_csv = write_status_csv(repo_root, clips)
    command_script = write_command_script(repo_root, commands)
    print_stage_command_map()
    print(f"\nWrote status CSV: {status_csv}")
    print(f"Wrote command script: {command_script}")
    print("\nHow to run:")
    print("  powershell -ExecutionPolicy Bypass -File {}".format(command_script))
    print("  powershell -ExecutionPolicy Bypass -File {} -WhatIf:$false".format(command_script))
    print_summary(clips)


if __name__ == "__main__":
    main()
