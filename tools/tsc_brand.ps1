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
  [string]$FontFile = "$PSScriptRoot\\..\\fonts\\Montserrat-ExtraBold.ttf",
  [string]$RibbonPNG,
  [string]$LogoPNG,
  [string]$HandleTxt
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
  $name = $name -replace '_WIDE_portrait_FINAL$', ''
  $name = $name -replace '_portrait_FINAL$', ''
  $name = $name -replace '_WIDE$', ''
  return $name
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

function New-BrandFilterGraph {
  param(
    [string]$BaseLabel = 'base',
    [string]$RibbonPath,
    [string]$LogoPath,
    [string]$HandleText,
    [int]$W = 1080,
    [int]$H = 1920
  )

  $inputs  = @()
  $filters = @()
  $filters += "[0:v]scale=${W}:${H}:flags=lanczos,setsar=1,setpts=PTS-STARTPTS[$BaseLabel]"

  $nextIn = 1
  $curLabel = $BaseLabel

  if ($RibbonPath -and (Test-Path $RibbonPath)) {
    $inputs += @('-i', $RibbonPath)
    $nextLabel = "ov$nextIn"
    $filters += "[$curLabel][$($nextIn):v]overlay=0:0:format=auto[$nextLabel]"
    $curLabel = $nextLabel
    $nextIn++
  }

  if ($LogoPath -and (Test-Path $LogoPath)) {
    $inputs += @('-i', $LogoPath)
    $nextLabel = "ov$nextIn"
    $filters += "[$curLabel][$($nextIn):v]overlay=32:main_h-overlay_h-32:format=auto[$nextLabel]"
    $curLabel = $nextLabel
    $nextIn++
  }

  if ($HandleText) {
    $escaped = Escape-DrawText $HandleText
    $nextLabel = "ov$nextIn"
    $filters += "[$curLabel]drawtext=text='$escaped':fontsize=44:fontcolor=white@0.95:borderw=2:bordercolor=black@0.55:x=36:y=H-th-36[$nextLabel]"
    $curLabel = $nextLabel
    $nextIn++
  }

  return @{
    Inputs    = $inputs
    FilterGraph = $filters
    OutLabel  = $curLabel
    NextIndex = $nextIn
  }
}

if (-not (Test-Path $In)) {
  throw "Input video not found: $In"
}

$inFull = (Resolve-Path -LiteralPath $In).Path
if (-not $Out) {
  $root = Get-PortraitRootFromPath -ReferencePath $inFull
  $base = Get-PortraitBasename -Path $inFull
  if ($root) {
    $Out = Join-Path (Join-Path $root 'branded') ($base + '_WIDE_portrait_FINAL.mp4')
  } else {
    $Out = Join-Path (Split-Path -Parent $inFull) ($base + '_WIDE_portrait_FINAL.mp4')
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

switch ($Aspect) {
  '16x9' { $targetWidth = 1920; $targetHeight = 1080 }
  '9x16' { $targetWidth = 1080; $targetHeight = 1920 }
}

$aspectAssets = @{}
$aspectAssets['title'] = if ($Aspect -eq '16x9') {
  Join-Path $brandPath 'title_ribbon_1920x1080.png'
} else {
  Join-Path $brandPath 'title_ribbon_1080x1920.png'
}
$aspectAssets['endcard'] = if ($Aspect -eq '16x9') {
  Join-Path $brandPath 'end_card_1920x1080.png'
} else {
  Join-Path $brandPath 'end_card_1080x1920.png'
}
$aspectAssets['guides'] = if ($Aspect -eq '16x9') {
  Join-Path $brandPath 'safe_guides_16x9.png'
} else {
  Join-Path $brandPath 'safe_guides_9x16.png'
}

$watermarkPath = Join-Path $brandPath 'watermark_corner_256.png'
$lowerThirdPath = Join-Path $brandPath 'lower_third_1920x220.png'

if (-not $PSBoundParameters.ContainsKey('RibbonPNG')) {
  if ($Title -or $Subtitle) {
    $RibbonPNG = $aspectAssets['title']
  }
}
if ($RibbonPNG -and -not (Test-Path $RibbonPNG)) {
  throw "Missing title ribbon asset: $RibbonPNG"
}

if (-not $PSBoundParameters.ContainsKey('LogoPNG')) {
  if ($Watermark) {
    $LogoPNG = $watermarkPath
  }
}
if (-not $Watermark) {
  $LogoPNG = $null
}
if ($LogoPNG -and -not (Test-Path $LogoPNG)) {
  throw "Missing watermark asset: $LogoPNG"
}

if ($ShowGuides -and -not (Test-Path $aspectAssets['guides'])) {
  throw "Missing safe guide asset for aspect $Aspect"
}
if ($EndCard -and -not (Test-Path $aspectAssets['endcard'])) {
  throw "Missing end card asset for aspect $Aspect"
}
if ($LowerThird -and -not (Test-Path $lowerThirdPath)) {
  throw "Missing lower third asset: $lowerThirdPath"
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

$brand = New-BrandFilterGraph -BaseLabel 'base' -RibbonPath $RibbonPNG -LogoPath $LogoPNG -HandleText $HandleTxt -W $targetWidth -H $targetHeight

$ffArgs = @('-hide_banner','-y','-i', $In)
if ($brand.Inputs.Count -gt 0) {
  $ffArgs += $brand.Inputs
}

$filterParts = @()
foreach ($part in @($brand.FilterGraph)) {
  if ($part) { $filterParts += $part }
}

$currentLabel = $brand.OutLabel
$nextIndex = $brand.NextIndex
if ($filterParts.Count -eq 0) {
  $filterParts += "[0:v]scale=${targetWidth}:${targetHeight}:flags=lanczos,setsar=1,setpts=PTS-STARTPTS[$currentLabel]"
  $currentLabel = 'base'
  $nextIndex = 1
}

if ($Title -and -not $fontMissing) {
  $titleText = Escape-DrawText $Title
  $titleFont = Get-DrawTextPath $FontFile
  $nextLabel = "ov$nextIndex"
  $filterParts += "[$currentLabel]drawtext=fontfile='$titleFont':text='$titleText':fontcolor=white:fontsize=72:x=260:y=80:shadowcolor=black@0.45:shadowx=2:shadowy=2[$nextLabel]"
  $currentLabel = $nextLabel
  $nextIndex++
}

if ($Subtitle -and -not $semiFontMissing) {
  $subtitleText = Escape-DrawText $Subtitle
  $subtitleFont = Get-DrawTextPath $semiFont
  $nextLabel = "ov$nextIndex"
  $filterParts += "[$currentLabel]drawtext=fontfile='$subtitleFont':text='$subtitleText':fontcolor=white:fontsize=48:x=260:y=150:shadowcolor=black@0.45:shadowx=2:shadowy=2[$nextLabel]"
  $currentLabel = $nextLabel
  $nextIndex++
}

if ($LowerThird) {
  $ffArgs += @('-i', $lowerThirdPath)
  $nextLabel = "ov$nextIndex"
  $filterParts += "[$currentLabel][$($nextIndex):v]overlay=x=(main_w-overlay_w)/2:y=main_h-overlay_h-120:format=auto[$nextLabel]"
  $currentLabel = $nextLabel
  $nextIndex++

  if ($Title -and -not $boldFontMissing) {
    $ltName = Escape-DrawText $Title
    $boldPath = Get-DrawTextPath $boldFont
    $nextLabel = "ov$nextIndex"
    $filterParts += "[$currentLabel]drawtext=fontfile='$boldPath':text='$ltName':fontcolor=white:fontsize=64:x=(w-1920)/2+220:y=h-220+48:shadowcolor=black@0.4:shadowx=2:shadowy=2[$nextLabel]"
    $currentLabel = $nextLabel
    $nextIndex++
  }
  if ($Subtitle -and -not $mediumFontMissing) {
    $ltRole = Escape-DrawText $Subtitle
    $mediumPath = Get-DrawTextPath $mediumFont
    $nextLabel = "ov$nextIndex"
    $filterParts += "[$currentLabel]drawtext=fontfile='$mediumPath':text='$ltRole':fontcolor=white:fontsize=40:x=(w-1920)/2+220:y=h-220+110:shadowcolor=black@0.4:shadowx=2:shadowy=2[$nextLabel]"
    $currentLabel = $nextLabel
    $nextIndex++
  }
}

if ($ShowGuides) {
  $ffArgs += @('-i', $aspectAssets['guides'])
  $nextLabel = "ov$nextIndex"
  $filterParts += "[$currentLabel][$($nextIndex):v]overlay=x=0:y=0:format=auto[$nextLabel]"
  $currentLabel = $nextLabel
  $nextIndex++
}

if ($EndCard) {
  $ffArgs += @('-loop','1','-t','2','-i', $aspectAssets['endcard'])
  $endIndex = $nextIndex
  $baseRef = "base_$endIndex"
  $filterParts += "[$($endIndex):v][$currentLabel]scale2ref=flags=lanczos[endcard$endIndex][$baseRef]"
  $filterParts += "[endcard$endIndex]trim=duration=2,setpts=PTS-STARTPTS[endcard_trim$endIndex]"
  $concatLabel = "ov$nextIndex"
  $filterParts += "[$baseRef][endcard_trim$endIndex]concat=n=2:v=1:a=0[$concatLabel]"
  $currentLabel = $concatLabel
  $nextIndex++
}

$finalLabel = "ov$nextIndex"
$filterParts += "[$currentLabel]format=yuv420p[$finalLabel]"
$currentLabel = $finalLabel
$nextIndex++

$filterComplex = $filterParts -join ';'

$bitrateStr = ($BitrateMbps.ToString('0.###', [System.Globalization.CultureInfo]::InvariantCulture)) + 'M'
$bufsizeStr = ($BitrateMbps * 2).ToString('0.###', [System.Globalization.CultureInfo]::InvariantCulture) + 'M'

$ffArgs += @(
  '-filter_complex', $filterComplex,
  '-map', "[$currentLabel]",
  '-map', '0:a?',
  '-c:v', 'libx264',
  '-profile:v', 'high',
  '-pix_fmt', 'yuv420p',
  '-b:v', $bitrateStr,
  '-maxrate', $bitrateStr,
  '-bufsize', $bufsizeStr,
  '-c:a', 'aac',
  '-b:a', '192k',
  '-ar', '48000',
  '-movflags', '+faststart',
  $Out
)

Write-Host "[brand] ffmpeg $($ffArgs -join ' ')"
& ffmpeg @ffArgs
if ($LASTEXITCODE -ne 0) {
  throw "Branding pass failed"
}

Write-Host "[brand] Created $Out"
