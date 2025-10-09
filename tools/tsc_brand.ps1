param(
  [Parameter(Mandatory=$true)] [string]$In,
  [string]$Out,
  [ValidateSet('16x9','9x16')] [string]$Aspect = '16x9',
  [string]$Title = '',
  [string]$Subtitle = '',
  [switch]$Watermark = $true,
  [switch]$LowerThird,
  [switch]$EndCard,
  [switch]$ShowGuides,
  [int]$BitrateMbps = 20,
  [string]$FontFile = "$PSScriptRoot\\..\\fonts\\Montserrat-ExtraBold.ttf"
)

$ErrorActionPreference = 'Stop'

function Ensure-ParentDirectory {
  param([string]$Path)
  if (-not $Path) { return }
  $parent = Split-Path -Parent ([System.IO.Path]::GetFullPath($Path))
  if ($parent -and -not (Test-Path $parent)) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
  }
}

function Get-PortraitRootFromPath {
  param([string]$ReferencePath)
  if (-not $ReferencePath) { return $null }
  $dir = if (Test-Path $ReferencePath -PathType Container) { $ReferencePath } else { Split-Path -Parent $ReferencePath }
  while ($dir) {
    if ((Split-Path $dir -Leaf) -ieq 'portrait_reels') { return $dir }
    $parent = Split-Path -Parent $dir
    if (-not $parent -or $parent -eq $dir) { break }
    $dir = $parent
  }
  return $null
}

function Get-PortraitBasename {
  param([string]$Path)
  if (-not $Path) { return 'reel' }
  $name = [System.IO.Path]::GetFileNameWithoutExtension($Path)
  if (-not $name) { return 'reel' }
  return ($name -replace '_portrait_FINAL$', '')
}

function Escape-DrawText {
  param([string]$Value)
  if (-not $Value) { return '' }
  $escaped = $Value -replace '\\', '\\\\'
  $escaped = $escaped -replace ':', '\\:'
  $escaped = $escaped -replace "'", "\\'"
  $escaped = $escaped -replace '%', '%%'
  return $escaped
}

function Get-DrawTextPath {
  param([string]$Path)
  $full = [System.IO.Path]::GetFullPath($Path)
  $forward = $full.Replace('\\', '/')
  if ($forward -match '^[A-Za-z]:/') {
    $forward = $forward.Insert(1, '\\')
  }
  return $forward
}

if (-not (Test-Path $In)) {
  throw "Input video not found: $In"
}

$inFull = (Resolve-Path -LiteralPath $In).Path
if (-not $Out) {
  $root = Get-PortraitRootFromPath -ReferencePath $inFull
  $base = Get-PortraitBasename -Path $inFull
  if ($root) {
    $Out = Join-Path (Join-Path $root 'branded') ($base + '_portrait_FINAL.mp4')
  } else {
    $Out = Join-Path (Split-Path -Parent $inFull) ($base + '_portrait_FINAL.mp4')
  }
}

$Out = [System.IO.Path]::GetFullPath($Out)
Ensure-ParentDirectory $Out

$ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpegCmd) {
  throw "ffmpeg not found on PATH"
}

$brandRoot = Join-Path $PSScriptRoot '..\brand\tsc'
$brandPath = (Resolve-Path -LiteralPath $brandRoot -ErrorAction Stop).Path

$aspectAssets = @{}
switch ($Aspect) {
  '16x9' {
    $aspectAssets['title'] = Join-Path $brandPath 'title_ribbon_1920x1080.png'
    $aspectAssets['endcard'] = Join-Path $brandPath 'end_card_1920x1080.png'
    $aspectAssets['guides'] = Join-Path $brandPath 'safe_guides_16x9.png'
  }
  '9x16' {
    $aspectAssets['title'] = Join-Path $brandPath 'title_ribbon_1080x1920.png'
    $aspectAssets['endcard'] = Join-Path $brandPath 'end_card_1080x1920.png'
    $aspectAssets['guides'] = Join-Path $brandPath 'safe_guides_9x16.png'
  }
}

$watermarkPath = Join-Path $brandPath 'watermark_corner_256.png'
$lowerThirdPath = Join-Path $brandPath 'lower_third_1920x220.png'

if (($Title -or $Subtitle) -and -not (Test-Path $aspectAssets['title'])) {
  throw "Missing title ribbon asset for aspect $Aspect"
}
if ($Watermark -and -not (Test-Path $watermarkPath)) {
  throw "Missing watermark asset: $watermarkPath"
}
if ($LowerThird -and -not (Test-Path $lowerThirdPath)) {
  throw "Missing lower third asset: $lowerThirdPath"
}
if ($ShowGuides -and -not (Test-Path $aspectAssets['guides'])) {
  throw "Missing safe guide asset for aspect $Aspect"
}
if ($EndCard -and -not (Test-Path $aspectAssets['endcard'])) {
  throw "Missing end card asset for aspect $Aspect"
}

$fontMissing = $false
if (-not (Test-Path $FontFile)) {
  Write-Warning "Font file not found: $FontFile (titles will render without text)"
  $fontMissing = $true
}

$fontDir = Split-Path -Parent ([System.IO.Path]::GetFullPath($FontFile))
$semiFont = if ($fontDir) { Join-Path $fontDir 'Montserrat-SemiBold.ttf' } else { $FontFile }
$boldFont = if ($fontDir) { Join-Path $fontDir 'Montserrat-Bold.ttf' } else { $FontFile }
$mediumFont = if ($fontDir) { Join-Path $fontDir 'Montserrat-Medium.ttf' } else { $FontFile }

$semiFontMissing = -not (Test-Path $semiFont)
$boldFontMissing = -not (Test-Path $boldFont)
$mediumFontMissing = -not (Test-Path $mediumFont)

if ($Subtitle -and $semiFontMissing) {
  Write-Warning "SemiBold font missing: $semiFont"
}
if ($LowerThird -and $boldFontMissing) {
  Write-Warning "Bold font missing for lower third: $boldFont"
}
if ($LowerThird -and $mediumFontMissing) {
  Write-Warning "Medium font missing for lower third: $mediumFont"
}

$ffArgs = @('-y', '-i', $In)
$filterParts = @('[0:v]setpts=PTS-STARTPTS,format=yuv420p[v0]')
$videoLabel = 'v0'
$inputIndex = 1

if ($Title -or $Subtitle) {
  $ffArgs += '-i'
  $ffArgs += $aspectAssets['title']
  $filterParts += "[$videoLabel][$inputIndex:v]overlay=x=0:y=0:format=auto[v$inputIndex]"
  $videoLabel = "v$inputIndex"
  $inputIndex++

  if ($Title -and -not $fontMissing) {
    $titleText = Escape-DrawText $Title
    $titleFont = Get-DrawTextPath $FontFile
    $filterParts += "[$videoLabel]drawtext=fontfile='$titleFont':text='$titleText':fontcolor=white:fontsize=72:x=260:y=80:shadowcolor=black@0.45:shadowx=2:shadowy=2[v$inputIndex]"
    $videoLabel = "v$inputIndex"
    $inputIndex++
  }
  if ($Subtitle -and -not $semiFontMissing) {
    $subtitleText = Escape-DrawText $Subtitle
    $subtitleFont = Get-DrawTextPath $semiFont
    $filterParts += "[$videoLabel]drawtext=fontfile='$subtitleFont':text='$subtitleText':fontcolor=white:fontsize=48:x=260:y=150:shadowcolor=black@0.45:shadowx=2:shadowy=2[v$inputIndex]"
    $videoLabel = "v$inputIndex"
    $inputIndex++
  }
}

if ($Watermark) {
  $ffArgs += '-i'
  $ffArgs += $watermarkPath
  $filterParts += "[$videoLabel][$inputIndex:v]overlay=x=32:y=main_h-overlay_h-32:format=auto[v$inputIndex]"
  $videoLabel = "v$inputIndex"
  $inputIndex++
}

if ($LowerThird) {
  $ffArgs += '-i'
  $ffArgs += $lowerThirdPath
  $filterParts += "[$videoLabel][$inputIndex:v]overlay=x=(main_w-overlay_w)/2:y=main_h-overlay_h-120:format=auto[v$inputIndex]"
  $videoLabel = "v$inputIndex"
  $inputIndex++

  if ($Title -and -not $boldFontMissing) {
    $ltName = Escape-DrawText $Title
    $boldPath = Get-DrawTextPath $boldFont
    $filterParts += "[$videoLabel]drawtext=fontfile='$boldPath':text='$ltName':fontcolor=white:fontsize=64:x=(w-1920)/2+220:y=h-220+48:shadowcolor=black@0.4:shadowx=2:shadowy=2[v$inputIndex]"
    $videoLabel = "v$inputIndex"
    $inputIndex++
  }
  if ($Subtitle -and -not $mediumFontMissing) {
    $ltRole = Escape-DrawText $Subtitle
    $mediumPath = Get-DrawTextPath $mediumFont
    $filterParts += "[$videoLabel]drawtext=fontfile='$mediumPath':text='$ltRole':fontcolor=white:fontsize=40:x=(w-1920)/2+220:y=h-220+110:shadowcolor=black@0.4:shadowx=2:shadowy=2[v$inputIndex]"
    $videoLabel = "v$inputIndex"
    $inputIndex++
  }
}

if ($ShowGuides) {
  $ffArgs += '-i'
  $ffArgs += $aspectAssets['guides']
  $filterParts += "[$videoLabel][$inputIndex:v]overlay=x=0:y=0:format=auto[v$inputIndex]"
  $videoLabel = "v$inputIndex"
  $inputIndex++
}

$finalLabel = $videoLabel

if ($EndCard) {
  $ffArgs += '-loop'; $ffArgs += '1'; $ffArgs += '-t'; $ffArgs += '2'; $ffArgs += '-i'; $ffArgs += $aspectAssets['endcard']
  $endIndex = $inputIndex
  $baseRef = "base_$endIndex"
  $filterParts += "[$endIndex:v][$finalLabel]scale2ref=flags=lanczos[endcard$endIndex][$baseRef]"
  $filterParts += "[endcard$endIndex]trim=duration=2,setpts=PTS-STARTPTS[endcard_trim$endIndex]"
  $filterParts += "[$baseRef][endcard_trim$endIndex]concat=n=2:v=1:a=0[vout]"
  $finalLabel = 'vout'
} else {
  $filterParts += "[$finalLabel]format=yuv420p[vout]"
  $finalLabel = 'vout'
}

$filterComplex = ($filterParts -join ';')

$bitrateStr = ($BitrateMbps.ToString('0.###', [System.Globalization.CultureInfo]::InvariantCulture)) + 'M'
$bufsizeStr = ($BitrateMbps * 2).ToString('0.###', [System.Globalization.CultureInfo]::InvariantCulture) + 'M'

$ffArgs += '-filter_complex'; $ffArgs += $filterComplex
$ffArgs += '-map'; $ffArgs += '[vout]'
$ffArgs += '-map'; $ffArgs += '0:a?'
$ffArgs += '-c:v'; $ffArgs += 'libx264'
$ffArgs += '-profile:v'; $ffArgs += 'high'
$ffArgs += '-pix_fmt'; $ffArgs += 'yuv420p'
$ffArgs += '-b:v'; $ffArgs += $bitrateStr
$ffArgs += '-maxrate'; $ffArgs += $bitrateStr
$ffArgs += '-bufsize'; $ffArgs += $bufsizeStr
$ffArgs += '-c:a'; $ffArgs += 'aac'
$ffArgs += '-b:a'; $ffArgs += '192k'
$ffArgs += '-ar'; $ffArgs += '48000'
$ffArgs += '-movflags'; $ffArgs += '+faststart'
$ffArgs += $Out

Write-Host "[brand] ffmpeg $($ffArgs -join ' ')"
& ffmpeg @ffArgs
if ($LASTEXITCODE -ne 0) {
  throw "Branding pass failed"
}

Write-Host "[brand] Created $Out"
