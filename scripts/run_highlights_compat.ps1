param(
  [string]$Video = ".\out\full_game_stabilized.mp4",
  [string]$OutDir = ".\out",
  [switch]$Reencode
)

$ErrorActionPreference = "Stop"
if (!(Test-Path $Video)) { throw "Missing input video: $Video" }
New-Item -ItemType Directory -Force -Path $OutDir, ".\scripts" | Out-Null

$env:PYTHONUNBUFFERED = "1"
$venv = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venv) { & $venv }

$logsDir  = Join-Path $OutDir "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

# Clean prior artifacts
@( "highlights.csv", "goal_resets.csv", "plays.csv", "segments.tsv",
   "concat_goals.txt", "concat_goals_plus_top.txt",
   "top_highlights_goals.mp4", "top_highlights_goals_first.mp4" ) |
  ForEach-Object { Remove-Item (Join-Path $OutDir $_) -ErrorAction SilentlyContinue }
Remove-Item (Join-Path $OutDir "clips_top\*"), (Join-Path $OutDir "clips_goals\*") -Recurse -ErrorAction SilentlyContinue

$hiCSV   = Join-Path $OutDir "highlights.csv"
$grCSV   = Join-Path $OutDir "goal_resets.csv"
$playsCSV = Join-Path $OutDir "plays.csv"

# 1) Detect events
& python -u .\02_detect_events.py `
  --video $Video `
  --out $hiCSV `
  --sample-fps 30 `
  --pass-window 2.0 `
  --passes-needed 3 `
  --pre 1.0 `
  --post 2.0 `
  --bias-blue 2>&1 | Tee-Object (Join-Path $logsDir "02_detect_events.log")
if ($LASTEXITCODE -ne 0) { throw "02_detect_events.py failed (see logs\\02_detect_events.log)" }

# 2) Goal resets (optional)
try {
  & python -u .\06_detect_goal_resets.py `
    --video $Video `
    --out $grCSV 2>&1 | Tee-Object (Join-Path $logsDir "06_detect_goal_resets.log")
  if ($LASTEXITCODE -ne 0) { throw "goal reset detector failed" }
} catch {
  Write-Warning "06_detect_goal_resets.py failed or produced no data; continuing without goals."
  Remove-Item $grCSV -ErrorAction SilentlyContinue
}

# 3) Rank actions with Navy bias
& python -u .\05_filter_by_motion.py `
  --video $Video `
  --csv $hiCSV `
  --out $playsCSV `
  --min-flow-mean 2.0 `
  --min-contig-frames 15 `
  --min-ball-speed 2.5 `
  --min-center-ratio 0.12 `
  --need-ball 1 `
  --navy-json .\config\team_navy.json `
  --team-hsv-low 100,25,25 `
  --team-hsv-high 140,255,255 `
  --rank-top 30 `
  --min-sep 4.0 `
  --audio-boost 1.0 2>&1 | Tee-Object (Join-Path $logsDir "05_filter_by_motion.log")
if ($LASTEXITCODE -ne 0) { throw "05_filter_by_motion.py failed (see logs\\05_filter_by_motion.log)" }

# 4) Build master reel (strict goals by default)
& .\scripts\make_reel.ps1 -Video $Video -OutDir $OutDir -GoalMode "strict" 2>&1 |
  Tee-Object (Join-Path $logsDir "make_reel.log")
if ($LASTEXITCODE -ne 0) { throw "make_reel.ps1 failed (see logs\\make_reel.log)" }

# For compatibility: optionally re-encode concat outputs
$concatGoals = Join-Path $OutDir "concat_goals.txt"
$concatBoth  = Join-Path $OutDir "concat_goals_plus_top.txt"
$outGoals = Join-Path $OutDir "top_highlights_goals.mp4"
$outBoth  = Join-Path $OutDir "top_highlights_goals_first.mp4"

function HasValidConcat($path) {
  Test-Path $path -and ((Get-Content $path | Where-Object { $_ -match '^file ' }).Count -gt 0)
}

if (HasValidConcat $concatGoals -and $Reencode) {
  & ffmpeg -f concat -safe 0 -i $concatGoals -r 24 -g 48 -c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p -c:a aac -ar 48000 -movflags +faststart $outGoals
}
if (HasValidConcat $concatGoals -and -not $Reencode -and -not (Test-Path $outGoals)) {
  & ffmpeg -f concat -safe 0 -i $concatGoals -c copy $outGoals
}

if (HasValidConcat $concatBoth -and $Reencode) {
  & ffmpeg -f concat -safe 0 -i $concatBoth -r 24 -g 48 -c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p -c:a aac -ar 48000 -movflags +faststart $outBoth
}
if (HasValidConcat $concatBoth -and -not $Reencode -and -not (Test-Path $outBoth)) {
  & ffmpeg -f concat -safe 0 -i $concatBoth -c copy $outBoth
}

Write-Host "[done]"
if (Test-Path $outGoals) { Write-Host ("  - " + $outGoals) }
if (Test-Path $outBoth)  { Write-Host ("  - " + $outBoth) }
if (-not (Test-Path $outGoals) -and -not (Test-Path $outBoth)) {
  Write-Host "  (No clips survived the filters -- check logs and adjust thresholds.)"
}
