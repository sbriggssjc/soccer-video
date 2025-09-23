param(
  [Parameter(Mandatory=$true)][string]$Input,
  [string]$Profile = "tiktok",
  [string]$Coeffs = "./zoom_coeffs.csv",
  [string]$Output = ""
)

if (-not (Test-Path $Input)) {
  throw "Input file not found: $Input"
}

if ([string]::IsNullOrWhiteSpace($Output)) {
  $stem = [System.IO.Path]::GetFileNameWithoutExtension($Input)
  $dir = Split-Path $Input -Parent
  if (-not $dir) { $dir = "." }
  $Output = Join-Path $dir "$stem.$Profile.zoom.mp4"
}

$cxExpr = "(in_w/2)"
$cyExpr = "(in_h/2)"
$zExpr  = "1.35"

$name = Split-Path $Input -Leaf
$row = $null
if (Test-Path $Coeffs) {
  $row = Import-Csv $Coeffs | Where-Object { $_.clip -eq $name -and $_.profile -eq $Profile } | Select-Object -First 1
}

if ($row) {
  $cxExpr = $row.cx_poly
  $cyExpr = $row.cy_poly
  $zExpr  = $row.z_poly
} else {
  # fallback: existing linear n-based expressions
  $cxExpr = "(in_w/2)"
  $cyExpr = "(in_h/2)"
  $zExpr  = "1.35"
}

switch ($Profile.ToLowerInvariant()) {
  "tiktok" {
    $targetW = 1080
    $targetH = 1920
  }
  default {
    $targetW = 1920
    $targetH = 1080
  }
}

$wExpr = "(in_w/($zExpr))"
$hExpr = "(in_h/($zExpr))"
$xExpr = "clip(($cxExpr)-(($wExpr)/2),0,in_w-($wExpr))"
$yExpr = "clip(($cyExpr)-(($hExpr)/2),0,in_h-($hExpr))"
$clip  = "crop=$wExpr:$hExpr:$xExpr:$yExpr"
$scale = "scale=$targetW:$targetH"
$setsar = "setsar=1"
$format = "format=yuv420p"

$vf = ($clip, $scale, $setsar, $format) -join ","

$ffmpegArgs = @(
  "-y",
  "-i", $Input,
  "-vf", $vf,
  "-an",
  "-c:v", "libx264",
  "-preset", "faster",
  "-crf", "18",
  $Output
)

Write-Host "Running: ffmpeg $($ffmpegArgs -join ' ')"
ffmpeg @ffmpegArgs
