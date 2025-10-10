param(
  [string]$Index = "out\events_selected_for_build.csv",
  [string]$Portrait = "1080x1920",
  [string]$Preset = "cinematic",
  [string]$LogsDir = "out\render_logs",
  [string]$BrandOverlay = "C:\Users\scott\soccer-video\brand\tsc\title_ribbon_1080x1920.png",
  [string]$EndCard = "C:\Users\scott\soccer-video\brand\tsc\end_card_1080x1920.png"
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Index)) { throw "Index not found: $Index" }

# Read the whole CSV text as-is (no parsing assumptions).
$csvText = Get-Content -LiteralPath $Index -Raw

# Find base IDs and explicit mp4s (keep it simple).
$rxBase = [regex]'\d{3}__[^\\/",\r\n]+__t\d+(?:\.\d+)?-t\d+(?:\.\d+)?'
$rxMp4  = [regex]'(?:[A-Za-z]:)?(?:[\\/][^"''\r\n]+)+\.mp4'

$ids  = $rxBase.Matches($csvText) | ForEach-Object { $_.Value } | Sort-Object -Unique
$mp4s = $rxMp4.Matches($csvText)  | ForEach-Object { $_.Value.Replace('/','\') } | Sort-Object -Unique

# Index all atomic clips (skip quarantine and copies).
$allFiles = Get-ChildItem -Path "out\atomic_clips" -Filter *.mp4 -Recurse -File |
  Where-Object { $_.FullName -notmatch '\\quarantine\\' -and $_.Name -notmatch '(?i)(__copy|_copy)\.mp4$' }

$byBase = @{}
foreach ($f in $allFiles) { $byBase[$f.BaseName] = $f.FullName }

# Resolve targets from IDs and explicit mp4s.
$resolved = New-Object System.Collections.Generic.HashSet[string]
foreach ($id in $ids) {
  if ($byBase.ContainsKey($id)) { [void]$resolved.Add($byBase[$id]) }
}
foreach ($p in $mp4s) {
  $cand = $p
  if (-not (Test-Path -LiteralPath $cand)) {
    $cand2 = Join-Path $PWD $p
    if (Test-Path -LiteralPath $cand2) { $cand = (Resolve-Path -LiteralPath $cand2).Path } else { continue }
  } else {
    $cand = (Resolve-Path -LiteralPath $cand).Path
  }
  if ($cand -notmatch '\\quarantine\\' -and $cand -notmatch '(?i)(__copy|_copy)\.mp4$') {
    [void]$resolved.Add($cand)
  }
}

$clips = @($resolved) | Sort-Object
if ($clips.Count -eq 0) { throw "No eligible clips after filtering." }

New-Item -ItemType Directory -Path $LogsDir -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path "out\portrait_reels\clean" -ErrorAction SilentlyContinue | Out-Null

foreach ($clip in $clips) {
  $base   = [IO.Path]::GetFileNameWithoutExtension($clip)
  $parent = Split-Path $clip -Parent

  # prefer locked path if present
  $ballPath = Join-Path $LogsDir ("{0}.ball.lock.jsonl" -f $base)
  if (-not (Test-Path $ballPath)) { $ballPath = Join-Path $LogsDir ("{0}.ball.jsonl" -f $base) }

  $tel   = Join-Path $LogsDir ("{0}.final.jsonl" -f $base)
  $log   = Join-Path $LogsDir ("{0}.final.log"  -f $base)
  $final = Join-Path "out\portrait_reels\clean" ("{0}_portrait_FINAL.mp4" -f $base)

  Write-Host "`n=== RENDER ==="
  Write-Host "IN : $clip"
  Write-Host "OUT: $final"

  $args = @("tools\render_follow_unified.py",
            "--in", $clip,
            "--preset", $Preset,
            "--portrait", $Portrait,
            "--clean-temp",
            "--lookahead","24",
            "--smoothing","0.65",
            "--zoom-min","1.08",
            "--zoom-max","1.45",
            "--speed-limit","280",
            "--telemetry", $tel,
            "--log", $log)

  if (Test-Path $ballPath) { $args += @("--ball-path", $ballPath, "--ball-key-x","bx_stab", "--ball-key-y","by_stab") }
  if (Test-Path $BrandOverlay) { $args += @("--brand-overlay", $BrandOverlay) }
  if (Test-Path $EndCard)      { $args += @("--endcard", $EndCard) }

  python $args

  # DEBUG overlay
  $dbg = Join-Path $parent ("{0}.__DEBUG_FINAL.mp4" -f $base)
  python tools\overlay_debug.py --in $clip --telemetry $tel --out $dbg
}

Write-Host "`nAll done."
