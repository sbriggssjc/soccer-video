param(
  [string]$Video = ".\out\full_game_stabilized.mp4",
  [string]$OutDir = ".\out",
  [switch]$Reencode,
  [switch]$UseTSCBrand,
  [string]$Title = ""
)

if (!(Test-Path $Video)) { throw "Missing input video: $Video" }
if (!(Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

$env:PYTHONUNBUFFERED = "1"
$venv = ".\.venv\Scripts\Activate.ps1"; if (Test-Path $venv) { & $venv }

$topDir   = Join-Path $OutDir "clips_top"
$goalsDir = Join-Path $OutDir "clips_goals"
$logsDir  = Join-Path $OutDir "logs"
New-Item -ItemType Directory -Force -Path $topDir,$goalsDir,$logsDir | Out-Null

Remove-Item `
  (Join-Path $OutDir "highlights.csv"),
  (Join-Path $OutDir "goal_resets.csv"),
  (Join-Path $OutDir "plays.csv"),
  (Join-Path $OutDir "concat_goals.txt"),
  (Join-Path $OutDir "concat_goals_plus_top.txt"),
  (Join-Path $OutDir "top_highlights_goals.mp4"),
  (Join-Path $OutDir "top_highlights_goals_first.mp4") `
  -ErrorAction SilentlyContinue
Remove-Item "$topDir\*", "$goalsDir\*" -Recurse -ErrorAction SilentlyContinue

# 1) Re-detect events (ball-in-play gated + action features)
python -u .\02_detect_events.py `
  --video $Video `
  --team navy `
  --gate-ball-in-play `
  --score action `
  --pre 1.0 --post 2.0 `
  --min-moving-players 4 `
  --out (Join-Path $OutDir "highlights.csv") `
  2>&1 | Tee-Object (Join-Path $logsDir "02_detect_events.log")

# 2) Goal resets
python -u .\06_detect_goal_resets.py `
  --video $Video `
  --out (Join-Path $OutDir "goal_resets.csv") `
  2>&1 | Tee-Object (Join-Path $logsDir "06_detect_goal_resets.log")

# 3) Action-focused filtering (fallback to motion if needed)
try {
  python -u .\05_filter_by_action.py `
    --video $Video `
    --highlights (Join-Path $OutDir "highlights.csv") `
    --goal-resets (Join-Path $OutDir "goal_resets.csv") `
    --team navy `
    --min-pass-chain 3 `
    --min-progression 12 `
    --include shots,attempts,buildups,switches,tackles `
    --top-dir $topDir `
    --goals-dir $goalsDir `
    2>&1 | Tee-Object (Join-Path $logsDir "05_filter_by_action.log")
}
catch {
  Write-Host "[warn] 05_filter_by_action.py missing; using motion fallback with stricter gates."
  python -u .\05_filter_by_motion.py `
    --video $Video `
    --highlights (Join-Path $OutDir "highlights.csv") `
    --goal-resets (Join-Path $OutDir "goal_resets.csv") `
    --team navy `
    --min-flow 2.2 --min-moving-players 4 --ball-on-pitch-required `
    --top-dir $topDir `
    --goals-dir $goalsDir `
    2>&1 | Tee-Object (Join-Path $logsDir "05_filter_by_motion.log")
}

# 4) Rebuild concat lists
(Get-ChildItem $goalsDir -Filter *.mp4 | Sort-Object Name | ForEach-Object { "file '$($_.FullName)'" }) `
  | Set-Content -Path (Join-Path $OutDir "concat_goals.txt")

@(
  (Get-ChildItem $goalsDir -Filter *.mp4 | Sort-Object Name | ForEach-Object { "file '$($_.FullName)'" })
  (Get-ChildItem $topDir   -Filter *.mp4 | Sort-Object Name | ForEach-Object { "file '$($_.FullName)'" })
) -join "`r`n" | Set-Content -Path (Join-Path $OutDir "concat_goals_plus_top.txt")

# 5) Stitch with ffmpeg (fix: pass flags as separate args)
$concatGoals = Join-Path $OutDir "concat_goals.txt"
$concatBoth  = Join-Path $OutDir "concat_goals_plus_top.txt"
$outGoals    = Join-Path $OutDir "top_highlights_goals.mp4"
$outBoth     = Join-Path $OutDir "top_highlights_goals_first.mp4"

if ($Reencode) {
  & ffmpeg -f concat -safe 0 -i $concatGoals -c:v libx264 -preset veryfast -crf 22 -c:a aac -movflags +faststart $outGoals
  & ffmpeg -f concat -safe 0 -i $concatBoth  -c:v libx264 -preset veryfast -crf 22 -c:a aac -movflags +faststart $outBoth
} else {
  & ffmpeg -f concat -safe 0 -i $concatGoals -c copy $outGoals
  & ffmpeg -f concat -safe 0 -i $concatBoth  -c copy $outBoth
}

if ($UseTSCBrand) {
  $brandScript = Join-Path $PSScriptRoot "..\tools\tsc_brand.ps1"
  if (-not (Test-Path $brandScript)) {
    throw "Brand script missing: $brandScript"
  }
  $branded = [System.IO.Path]::ChangeExtension($outBoth, '.tsc.mp4')
  & $brandScript -In $outBoth -Out $branded -Title $Title -Watermark -EndCard -Aspect '16x9'
  if ($LASTEXITCODE -ne 0) { throw "Brand pass failed." }
  Move-Item $branded $outBoth -Force
}

Write-Host "[done] Rebuilt highlights:"
Write-Host "  - $outGoals"
Write-Host "  - $outBoth"
