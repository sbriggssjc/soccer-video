[CmdletBinding()]
param(
  [string]$Index = "out\events_selected_for_build.csv",
  [string]$Portrait = "1080x1920",
  [string]$Preset = "cinematic",
  [string]$LogsDir = "out\render_logs",
  [string]$BrandOverlay = "C:\Users\scott\soccer-video\brand\tsc\title_ribbon_1080x1920.png",
  [string]$EndCard = "C:\Users\scott\soccer-video\brand\tsc\end_card_1080x1920.png",
  [switch]$ListOnly  # just list what would run
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Index)) { throw "Index not found: $Index" }

Write-Host "Using index: $Index"

# --- Collect base IDs & explicit mp4 paths from BOTH raw text and CSV rows ---
$rxBase = [regex]'\d{3}__[^\\/",\r\n]+__t\d+(?:\.\d+)?-t\d+(?:\.\d+)?'
$rxMp4  = [regex]'(?:[A-Za-z]:)?(?:[\\/][^"''\r\n]+)+\.mp4'

$ids  = New-Object System.Collections.Generic.HashSet[string]
$mp4s = New-Object System.Collections.Generic.HashSet[string]

# a) scan raw text (handles arbitrary CSV)
$raw = Get-Content -LiteralPath $Index -Raw -Encoding UTF8
foreach ($m in $rxBase.Matches($raw)) { [void]$ids.Add($m.Value) }
foreach ($m in $rxMp4.Matches($raw))  { [void]$mp4s.Add($m.Value.Replace('/','\')) }

# b) scan parsed CSV (covers nicely-named columns like "clip","path", etc.)
try {
  $rows = Import-Csv -LiteralPath $Index
  foreach ($r in $rows) {
    foreach ($prop in $r.PSObject.Properties) {
      $val = [string]$prop.Value
      if ([string]::IsNullOrWhiteSpace($val)) { continue }
      foreach ($m in $rxBase.Matches($val)) { [void]$ids.Add($m.Value) }
      foreach ($m in $rxMp4.Matches($val))  { [void]$mp4s.Add($m.Value.Replace('/','\')) }
    }
  }
} catch {
  Write-Verbose "Import-Csv failed (non-fatal): $($_.Exception.Message)"
}

# --- Index all atomic clips on disk (skip quarantine & copies) ---
$allFiles = Get-ChildItem -Path "out\atomic_clips" -Filter *.mp4 -Recurse -File |
  Where-Object {
    $_.FullName -notmatch '\\quarantine\\' -and
    $_.Name -notmatch '(?i)(__copy|_copy)\.mp4$'
  }

$byBase = @{}
foreach ($f in $allFiles) { $byBase[$f.BaseName] = $f.FullName }

# --- Resolve targets from IDs + explicit mp4 paths ---
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

# --- Diagnostics so you can see what's happening ---
Write-Host "Found on disk under out\atomic_clips : $($allFiles.Count) mp4s (after quarantine/copy filter)."
Write-Host "IDs parsed from index                 : $($ids.Count)"
Write-Host "Explicit mp4 paths parsed from index  : $($mp4s.Count)"
Write-Host "Resolved runnable clips               : $($clips.Count)"
if ($ids.Count -gt 0)   { Write-Host "Example ID(s): $((@($ids)[0..([Math]::Min(4,$ids.Count-1))]) -join ', ')" }
if ($clips.Count -gt 0) { Write-Host "First clip: $($clips[0])" }

if ($clips.Count -eq 0) {
  Write-Warning "Nothing matched. Quick checklist:"
  Write-Host "  • Does your CSV contain either base IDs like 022__SHOT__t3028.10-t3059.70 OR full mp4 paths?"
  Write-Host "  • Do the files exist under out\atomic_clips (and not in \quarantine\ or named *_copy.mp4)?"
  Write-Host "  • If your CSV uses a different ID pattern, send me one example row."
  throw "No eligible clips after filtering."
}

if ($ListOnly) {
  Write-Host "`n[ListOnly] Would render these $($clips.Count) clips:"
  $clips | ForEach-Object { Write-Host "  - $_" }
  return
}

# --- Make sure output dirs exist ---
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
