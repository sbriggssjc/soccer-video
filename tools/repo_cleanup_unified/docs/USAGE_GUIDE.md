# RepoClean Usage Guide

This guide is the task-focused companion to the main README. Each section includes copy/paste-ready commands and highlights the outputs you should inspect before making decisions.

## Inventory and duplicate triage
Use Inventory first to build the dataset of files, metadata, and hash groups. `-Fast` performs selective hashing using size/duration buckets; remove it for a full checksum sweep.

```powershell
$Root = "C:\\Users\\scott\\soccer-video"

# Build inventory with selective hashing
powershell -ExecutionPolicy Bypass -File tools\repo_cleanup_unified\RepoClean.ps1 -Mode Inventory -Root $Root -Fast

# Full checksum inventory (slower but comprehensive)
powershell -ExecutionPolicy Bypass -File tools\repo_cleanup_unified\RepoClean.ps1 -Mode Inventory -Root $Root
```

Review the following outputs under `out\inventory`:

- `repo_inventory.csv/json` – full manifest with size, timestamps, optional duration/resolution, and status classification (KEEP / KEEP_SIDECAR / CANDIDATE_REMOVE / ORPHAN).
- `duplicates_exact.csv` – assets sharing an identical hash.
- `duplicates_probable.csv` – size/duration/resolution cohorts that need manual review.
- `keep_candidates.csv`, `remove_candidates.csv`, `orphans.csv` – quick filters derived from `KeepRules.psd1`.
- `existing_tools.csv` and `docs\SCRIPT_COMPAT_MATRIX.md` – legacy script discovery and mapping.

## Building a cleanup plan
Generate a plan after reviewing inventory outputs. Plans default to the **Quarantine** strategy (safe mirroring into `_quarantine`).

```powershell
# Default quarantine plan
powershell -ExecutionPolicy Bypass -File tools\repo_cleanup_unified\RepoClean.ps1 -Mode Plan -Root $Root

# Plan with NTFS hardlink dedupe enabled
powershell -ExecutionPolicy Bypass -File tools\repo_cleanup_unified\RepoClean.ps1 -Mode Plan -Root $Root -EnableHardlink

# Delete strategy (Recycle Bin). Add -Permanent to bypass the bin.
powershell -ExecutionPolicy Bypass -File tools\repo_cleanup_unified\RepoClean.ps1 -Mode Plan -Root $Root -Strategy Delete
```

Plan artifacts live in `out\plans`:

- `cleanup_plan.csv` – actionable entries with action, target/quarantine path, and rationale.
- `hardlink_plan.csv` – present only when `-EnableHardlink` is used and duplicates were found.
- `human_readable_summary.txt` – top folders by potential savings plus a high-level overview.

Plans include the repo root signature and expire after 48 hours to prevent stale execution.

## Executing a plan
Execution is opt-in. You must supply both `-ConfirmRun` and `-DryRun:$false`. Without these flags the script exits immediately.

```powershell
# Execute the most recent plan (requires prior Plan run)
powershell -ExecutionPolicy Bypass -File tools\repo_cleanup_unified\RepoClean.ps1 -Mode Execute -Root $Root -ConfirmRun -DryRun:$false
```

Execution safeguards:

- Rejects plans older than 48 hours or produced for a different `-Root`.
- Preserves original timestamps when quarantining files.
- Uses Windows Recycle Bin for delete strategy when available; otherwise falls back to permanent removal or quarantine as configured.
- Logs every action to `out\logs\repo_cleanup_run_*.log` and writes failures to `failed_actions.csv`.

## Hardlink dedupe follow-up
If you enabled `-EnableHardlink`, inspect `out\plans\hardlink_plan.csv` to validate source/target pairings. RepoClean does **not** execute hardlink changes automatically; perform them manually or script against the plan to ensure the canonical copy is correct.

## Orphan review workflow
Orphans are files that do not match any keep or remove glob. Triage them before deleting:

1. Inspect `out\inventory\orphans.csv` for patterns or false negatives.
2. If a file should be preserved long term, add a glob to `KeepRules.psd1` (and rerun Inventory).
3. For throwaway intermediates, allow the plan to quarantine/delete them or add a targeted pattern to `RemoveGlobs` for future automation.

## Refreshing the season index
Inventory must be up to date before running Index mode. The index summarises kept masters, atomics, and portrait reels by top-level bucket (typically season or game).

```powershell
powershell -ExecutionPolicy Bypass -File tools\repo_cleanup_unified\RepoClean.ps1 -Mode Index -Root $Root
```

Outputs: `out\index\season_index.csv` and `.json`, including per-bucket counts, total atomic duration, and a `MissingReel` flag when final reels are absent but atomic clips exist.

## Environment diagnostics
Run Doctor to validate prerequisites before a large cleanup session.

```powershell
powershell -ExecutionPolicy Bypass -File tools\repo_cleanup_unified\RepoClean.ps1 -Mode Doctor -Root $Root
```

Doctor checks:

- `ffprobe` availability for media metadata enrichment.
- PowerShell version (PS 7+ enables parallel execution; PS 5.1 falls back to sequential processing).
- NTFS hardlink compatibility and Windows long-path settings when available.
- Writes a human-readable report to `out\inventory\environment_report.txt`.

## Custom rules workflow
When you edit `KeepRules.psd1`, rerun Inventory so the new globs classify files correctly. The hash cache persists between runs (`out\inventory\hash_cache.csv`), so unchanged files avoid re-hashing.

## Troubleshooting tips
- If `ffprobe` is missing, install `ffmpeg` or place `ffprobe.exe` in your PATH; RepoClean will skip duration/resolution metadata otherwise.
- On Windows PowerShell 5.1, expect sequential execution and slower hashing. Upgrade to PowerShell 7+ for `ForEach-Object -Parallel` support.
- Ensure `_quarantine` resides on the same volume as the source tree when using hardlinks; NTFS hardlinks cannot span volumes.
- Review `docs\SCRIPT_COMPAT_MATRIX.md` to confirm which legacy scripts can be retired after adopting RepoClean.
