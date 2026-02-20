param(
  [int]    $Scale   = 2,
  [string] $OutDir,
  # Optional: pass your overlay/endcard (leave blank to omit)
  [string] $BrandOverlay,
  [string] $Endcard
)

$RepoRoot  = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $OutDir)       { $OutDir       = Join-Path $RepoRoot "out\portrait_reels\clean" }
if (-not $BrandOverlay) { $BrandOverlay = Join-Path $RepoRoot "brand\tsc\title_ribbon_1080x1920.png" }
if (-not $Endcard)      { $Endcard      = Join-Path $RepoRoot "brand\tsc\end_card_1080x1920.png" }
$AtomicDir = Join-Path $RepoRoot "out\atomic_clips"
$Renderer  = Join-Path $RepoRoot "tools\render_follow_unified.py"

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force $OutDir | Out-Null }

# 🔎 RECURSIVE scan; ignore MASTER.* and anything inside _quarantine
$clips = Get-ChildItem $AtomicDir -Recurse -File -Include *.mp4,*.mov |
         Where-Object {
           $_.Name -notmatch '^(MASTER|master)\.(mp4|mov)$' -and
           $_.FullName -notmatch '\\_quarantine(\\|$)'
         } |
         Sort-Object FullName

if ($clips.Count -eq 0) {
  Write-Warning "No staged atomic clips found under $AtomicDir (recursively)."
  return
}

function Invoke-Render {
  param([string]$In, [string]$OutPath)

  $args = @(
    $Renderer, "--in", $In,
    "--portrait", "1080x1920",
    "--upscale", "--upscale-scale", $Scale.ToString(),
    "--out", $OutPath,
    "--preset", "cinematic"
  )
  if ($BrandOverlay -and (Test-Path $BrandOverlay)) { $args += @("--brand-overlay", $BrandOverlay) }
  if ($Endcard -and (Test-Path $Endcard))           { $args += @("--endcard",      $Endcard) }

  & python $args
  return $LASTEXITCODE
}

$done = 0; $skipped = 0; $failed = 0

foreach ($c in $clips) {
  $outName = $c.BaseName + "_WIDE_portrait_FINAL.mp4"
  $outPath = Join-Path $OutDir $outName

  $need = $true
  if (Test-Path $outPath) {
    $need = -not ((Get-Item $outPath).LastWriteTime -gt (Get-Item $c.FullName).LastWriteTime)
  }
  if (-not $need) { $skipped++; continue }

  Write-Host "Processing $($c.FullName)..."
  $rc = Invoke-Render -In $c.FullName -OutPath $outPath
  if ($rc -eq 0 -and (Test-Path $outPath)) { $done++ } else { $failed++ }
}

Write-Host "Batch complete. Done: $done  Skipped: $skipped  Failed: $failed"
