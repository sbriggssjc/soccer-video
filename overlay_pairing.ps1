param(
  [Parameter(Mandatory = $true)]
  [string]$Stem,  # e.g. 004__PRESSURE__t640.00-t663.00

  [ValidateSet("final","source")]
  [string]$Mode = "final"  # "final" = portrait + final.jsonl, "source" = atomic + ball.jsonl
)

$ProjRoot  = "C:\Users\scott\soccer-video"
$AtomicDir = Join-Path $ProjRoot "out\atomic_clips\2025-10-12__TSC_SLSG_FallFestival"
$OutDir    = Join-Path $ProjRoot "out\portrait_reels\clean"
$LogsDir   = Join-Path $ProjRoot "out\render_logs"

function Require-File([string]$p, [string]$hint) {
  if (-not $p -or -not (Test-Path $p -PathType Leaf)) {
    throw "Missing file: $hint`nExpected at: $p"
  }
}

# --- choose the correct pair (video, telemetry) ---
if ($Mode -eq "final") {
  $Video = Join-Path $OutDir  ("{0}_portrait_FINAL.mp4" -f $Stem)
  $Telem = Join-Path $LogsDir ("{0}.final.jsonl"       -f $Stem)
  Require-File $Video "Final portrait mp4"
  Require-File $Telem "Final telemetry jsonl"
}
else {
  $Video = Join-Path $AtomicDir ("{0}.mp4" -f $Stem)
  $Plan1 = Join-Path $LogsDir   ("{0}.ball.lock.jsonl" -f $Stem)
  $Plan2 = Join-Path $LogsDir   ("{0}.ball.jsonl"      -f $Stem)

  if (Test-Path $Plan1 -PathType Leaf) { $Telem = $Plan1 }
  elseif (Test-Path $Plan2 -PathType Leaf) { $Telem = $Plan2 }
  else { throw "Missing file: Ball plan jsonl`nExpected at: $Plan1 OR $Plan2" }

  Require-File $Video "Atomic source mp4"
}

Write-Host "Video    : $Video"   -ForegroundColor Cyan
Write-Host "Telemetry: $Telem"   -ForegroundColor Cyan

# --- dimensions (video) ---
$vh = & ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$Video"
if (-not $vh) { throw "ffprobe couldn't read: $Video" }
$w,$h = $vh.Split(',')

# --- telemetry ranges + NaNs ---
$bxMin=[double]::PositiveInfinity; $byMin=[double]::PositiveInfinity
$bxMax=[double]::NegativeInfinity; $byMax=[double]::NegativeInfinity
$nanCount = 0; $rowCount = 0

Get-Content "$Telem" | ForEach-Object {
  if ($_ -and $_.Trim()) {
    $rowCount++
    $o = $_ | ConvertFrom-Json
    $bx = if ($o.PSObject.Properties.Name -contains 'bx_stab') { $o.bx_stab } else { $o.bx }
    $by = if ($o.PSObject.Properties.Name -contains 'by_stab') { $o.by_stab } else { $o.by }

    $bxIsNan = ($bx -ne $null) -and [double]::IsNaN([double]$bx)
    $byIsNan = ($by -ne $null) -and [double]::IsNaN([double]$by)
    if ($bx -eq $null -or $by -eq $null -or $bxIsNan -or $byIsNan) { $nanCount++; return }

    if ($bx -lt $bxMin) { $bxMin = $bx }; if ($bx -gt $bxMax) { $bxMax = $bx }
    if ($by -lt $byMin) { $byMin = $by }; if ($by -gt $byMax) { $byMax = $by }
  }
}

Write-Host ("Video size: {0}x{1}" -f $w,$h) -ForegroundColor Green
Write-Host ("bx range : {0:N1} → {1:N1}" -f $bxMin,$bxMax) -ForegroundColor Green
Write-Host ("by range : {0:N1} → {1:N1}" -f $byMin,$byMax) -ForegroundColor Green
Write-Host ("rows     : {0}, NaNs: {1}" -f $rowCount,$nanCount) -ForegroundColor Green

# warn if coordinate space mismatches the video dimensions badly
if ( ($bxMax -gt ([double]$w*1.2)) -or ($byMax -gt ([double]$h*1.2)) ) {
  Write-Warning "Telemetry coords exceed video frame — wrong pairing (use Mode='final' for portrait + final.jsonl or Mode='source' for atomic + ball.jsonl)."
}

# --- prep telemetry for overlay_debug (ensure bx/by exist, forward-fill if gaps) ---
$TelemForOverlay = ($Telem -replace '\.jsonl$','__overlay.jsonl')
$lastBx=$null; $lastBy=$null
Get-Content "$Telem" | ForEach-Object {
  if ($_ -and $_.Trim()) {
    $o = $_ | ConvertFrom-Json
    $bx = if ($o.PSObject.Properties.Name -contains 'bx_stab') { $o.bx_stab } else { $o.bx }
    $by = if ($o.PSObject.Properties.Name -contains 'by_stab') { $o.by_stab } else { $o.by }
    if ($bx -eq $null -or [double]::IsNaN([double]$bx)) { $bx = $lastBx }
    if ($by -eq $null -or [double]::IsNaN([double]$by)) { $by = $lastBy }
    if ($bx -ne $null -and $by -ne $null) { $lastBx=$bx; $lastBy=$by }
    $o | Add-Member -Force -NotePropertyName bx -NotePropertyValue $bx
    $o | Add-Member -Force -NotePropertyName by -NotePropertyValue $by
    ($o | ConvertTo-Json -Compress)
  }
} | Set-Content -Encoding utf8 $TelemForOverlay

# --- overlay next to chosen video ---
$OutOverlay = [IO.Path]::Combine([IO.Path]::GetDirectoryName($Video), ($Stem + ".__DEBUG_OVERLAY.mp4"))
python tools\overlay_debug.py --in "$Video" --telemetry "$TelemForOverlay" --out "$OutOverlay"
Write-Host "Overlay written: $OutOverlay" -ForegroundColor Magenta
