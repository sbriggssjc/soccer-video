# RepoClean Unified Maintenance Tool

RepoClean consolidates the historical cleanup, dedupe, indexing, and diagnostic scripts that were scattered across the soccer-video repository into one production-grade PowerShell entry point. It is designed to inventory the workspace safely, plan space-saving actions, execute those plans with explicit confirmation, and build up-to-date indices for season deliverables without sacrificing the masters, canonical atomic clips, or polished reels.

## Why unify?
- **Single source of truth** – replace dozens of overlapping scripts with one supported tool and one configurable ruleset.
- **Safety-first execution** – every destructive action must go through Inventory → Plan → explicit Execute confirmation. Plans age out after 48 hours.
- **Future-proof tuning** – day-to-day exceptions live in `KeepRules.psd1` so you can evolve glob coverage without editing code.
- **Cross-mode visibility** – Inventory emits duplicates, keep/remove/orphan lists, and compatibility reports for older scripts.

## What RepoClean preserves
RepoClean protects three pillars of the post-production workflow:

1. **Masters** – ordered master videos, season/game indices, and master logs (CSV/JSON).
2. **Atomic clips** – canonical per-event source clips that feed downstream reels.
3. **Polished portrait reels** – final “postable” portrait exports with ball-follow tracking.

Sidecars (`.csv`, `.json`, `.srt`, `.vtt`, `.txt`) with matching basenames ride along with any kept media automatically. Everything else becomes a candidate for quarantine, dedupe, or removal depending on the selected strategy.

## Quick start
```powershell
# Go to repo root
$Root = "C:\\Users\\scott\\soccer-video"

# 0) Doctor (optional)
powershell -ExecutionPolicy Bypass -File tools\repo_cleanup_unified\RepoClean.ps1 -Mode Doctor -Root $Root

# 1) Inventory (safe)
powershell -ExecutionPolicy Bypass -File tools\repo_cleanup_unified\RepoClean.ps1 -Mode Inventory -Root $Root -Fast

# 2) Plan (safe) – default Strategy=Quarantine
powershell -ExecutionPolicy Bypass -File tools\repo_cleanup_unified\RepoClean.ps1 -Mode Plan -Root $Root

# 3) Review outputs under out\inventory and out\plans

# 4) Execute (WILL ACT ONLY IF you pass both flags)
powershell -ExecutionPolicy Bypass -File tools\repo_cleanup_unified\RepoClean.ps1 -Mode Execute -Root $Root -ConfirmRun -DryRun:$false

# 5) Build/refresh season index (safe)
powershell -ExecutionPolicy Bypass -File tools\repo_cleanup_unified\RepoClean.ps1 -Mode Index -Root $Root
```

## Output structure
All generated data lives under the repository’s `out` directory to avoid polluting source folders.

- `out/inventory` – inventory CSV/JSON, duplicate reports, keep/remove/orphan lists, atomic → master maps, environment reports, `existing_tools.csv`, and the rolling `hash_cache.csv`.
- `out/plans` – `cleanup_plan.csv`, optional `hardlink_plan.csv`, and a human-readable summary.
- `out/logs` – time-stamped execution logs and `failed_actions.csv`.
- `out/index` – refreshed season rollups (`season_index.csv/json`).

## Safety model & recovery
- Inventory and Plan modes never move or delete files.
- Execute mode is gated by **both** `-ConfirmRun` and `-DryRun:$false`. Plans older than 48 hours or generated for a different root are rejected.
- Default strategy moves artifacts into a mirrored `_quarantine` tree (timestamps preserved). Restore by moving items back into place and rerunning Inventory.
- Delete strategy sends files to the Windows Recycle Bin when available, unless `-Permanent` is specified.

## Customising keep/remove rules
All classification logic lives in [`KeepRules.psd1`](../KeepRules.psd1). Update glob lists to add new master folders, reel destinations, or junk patterns without touching the script. Sidecar extensions automatically follow any kept asset.

## Compatibility map
`docs/SCRIPT_COMPAT_MATRIX.md` is regenerated automatically whenever Inventory or Doctor mode runs. It lists the legacy scripts discovered across the repo and which RepoClean subcommand supersedes them. The first Inventory pass also records the scan details in `out/inventory/existing_tools.csv` for auditing.

## Change tracking
See [`docs/CHANGELOG.md`](CHANGELOG.md) for the evolving release notes. The initial entry documents the migration into RepoClean and references the compatibility matrix so future updates can append entries without editing code paths.
