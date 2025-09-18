param(
  [string]$Video = ".\out\full_game_stabilized.mp4",
  [string]$OutDir = ".\out",
  [switch]$Reencode
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $Video)) { throw "Missing input video: $Video" }
New-Item -ItemType Directory -Force -Path $OutDir, ".\scripts" | Out-Null

$env:PYTHONUNBUFFERED = "1"
$venv = ".\.venv\Scripts\Activate.ps1"; if (Test-Path $venv) { & $venv }

$topDir   = Join-Path $OutDir "clips_top"
$goalsDir = Join-Path $OutDir "clips_goals"
$logsDir  = Join-Path $OutDir "logs"
New-Item -ItemType Directory -Force -Path $topDir,$goalsDir,$logsDir | Out-Null

# Clean prior artifacts
@("highlights.csv","goal_resets.csv","plays.csv",
  "concat_goals.txt","concat_goals_plus_top.txt",
  "top_highlights_goals.mp4","top_highlights_goals_first.mp4") |
  ForEach-Object { Remove-Item (Join-Path $OutDir $_) -ErrorAction SilentlyContinue }
Remove-Item "$topDir\*", "$goalsDir\*" -Recurse -ErrorAction SilentlyContinue

# 1) Detect events (legacy args; bias to Navy/blue; tighter pre/post)
$hiCSV = Join-Path $OutDir "highlights.csv"
& python -u .\02_detect_events.py `
  --video $Video `
  --out $hiCSV `
  --sample-fps 30 `
  --pass-window 2.0 `
  --passes-needed 3 `
  --pre 1.0 --post 2.0 `
  --bias-blue 2>&1 | Tee-Object (Join-Path $logsDir "02_detect_events.log")
if ($LASTEXITCODE -ne 0) { throw "02_detect_events.py failed (see logs\02_detect_events.log)" }

# 2) Goal resets
$grCSV = Join-Path $OutDir "goal_resets.csv"
& python -u .\06_detect_goal_resets.py `
  --video $Video `
  --out $grCSV 2>&1 | Tee-Object (Join-Path $logsDir "06_detect_goal_resets.log")
if ($LASTEXITCODE -ne 0) { throw "06_detect_goal_resets.py failed (see logs\06_detect_goal_resets.log)" }

# 3) Motion filter with stricter gates (approx action-only)
& python -u .\05_filter_by_motion.py `
  --video $Video `
  --highlights $hiCSV `
  --goal-resets $grCSV `
  --min-flow 2.2 `
  --min-moving-players 4 `
  --ball-on-pitch-required `
  --top-dir $topDir `
  --goals-dir $goalsDir 2>&1 | Tee-Object (Join-Path $logsDir "05_filter_by_motion.log")
if ($LASTEXITCODE -ne 0) { throw "05_filter_by_motion.py failed (see logs\05_filter_by_motion.log)" }

# 4) Rebuild concat lists only if clips exist (ASCII to satisfy ffmpeg)
$goalClips = Get-ChildItem $goalsDir -Filter *.mp4 -ErrorAction SilentlyContinue | Sort-Object Name
$topClips  = Get-ChildItem $topDir   -Filter *.mp4 -ErrorAction SilentlyContinue | Sort-Object Name

$concatGoals = Join-Path $OutDir "concat_goals.txt"
$concatBoth  = Join-Path $OutDir "concat_goals_plus_top.txt"

if ($goalClips.Count -gt 0) {
  $goalLines = $goalClips | ForEach-Object { "file '$($_.FullName)'" }
  Set-Content -Path $concatGoals -Value $goalLines -Encoding ascii
}

if ($goalClips.Count -gt 0 -or $topClips.Count -gt 0) {
  $bothLines = @()
  $bothLines += ($goalClips | ForEach-Object { "file '$($_.FullName)'" })
  $bothLines += ($topClips  | ForEach-Object { "file '$($_.FullName)'" })
  Set-Content -Path $concatBoth -Value $bothLines -Encoding ascii
}

function HasValidConcat($path) {
  Test-Path $path -and ((Get-Content $path | Where-Object { $_ -match '^file ' }).Count -gt 0)
}

# 5) Stitch with ffmpeg only when lists are valid
$outGoals = Join-Path $OutDir "top_highlights_goals.mp4"
$outBoth  = Join-Path $OutDir "top_highlights_goals_first.mp4"

if (HasValidConcat $concatGoals) {
  if ($Reencode) {
    & ffmpeg -f concat -safe 0 -i $concatGoals -c:v libx264 -preset veryfast -crf 22 -c:a aac -movflags +faststart $outGoals
  } else {
    & ffmpeg -f concat -safe 0 -i $concatGoals -c copy $outGoals
  }
}

if (HasValidConcat $concatBoth) {
  if ($Reencode) {
    & ffmpeg -f concat -safe 0 -i $concatBoth -c:v libx264 -preset veryfast -crf 22 -c:a aac -movflags +faststart $outBoth
  } else {
    & ffmpeg -f concat -safe 0 -i $concatBoth -c copy $outBoth
  }
}

Write-Host "[done]"
if (Test-Path $outGoals) { Write-Host ("  - " + $outGoals) }
if (Test-Path $outBoth)  { Write-Host ("  - " + $outBoth) }
if (-not (Test-Path $outGoals) -and -not (Test-Path $outBoth)) {
  Write-Host "  (No clips survived the stricter filters -- check logs and adjust thresholds.)"
}
