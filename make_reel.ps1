param(
    [Parameter(Mandatory=$true)][string]$Input,
    [Parameter(Mandatory=$true)][string]$Vars,
    [string]$Output = "",
    [string]$Debug = "",
    [string]$Compare = "",
    [ValidateSet("portrait","landscape")][string]$Profile = "portrait",
    [double]$Crf = 18,
    [string]$Preset = "slow",
    [double]$DebugCrf = 20,
    [string]$DebugPreset = "veryfast",
    [double]$CompareCrf = 20,
    [string]$ComparePreset = "veryfast"
)

$ErrorActionPreference = "Stop"
if (!(Test-Path $Input)) { throw "Missing input: $Input" }
if (!(Test-Path $Vars)) { throw "Missing vars file: $Vars" }
$inputFull = (Resolve-Path -LiteralPath $Input).Path
$varsFull = (Resolve-Path -LiteralPath $Vars).Path

function Get-PortraitBaseName {
    param(
        [string]$InputPath,
        [string]$VarsPath
    )

    $base = ""
    if ($VarsPath) {
        $base = [IO.Path]::GetFileNameWithoutExtension($VarsPath)
    }
    if (-not $base -and $InputPath) {
        $base = [IO.Path]::GetFileNameWithoutExtension($InputPath)
    }
    if (-not $base) {
        $base = "reel"
    }
    return ($base -replace "_portrait_FINAL$", "")
}

function Ensure-ParentDirectory {
    param([string]$Path)
    if (-not $Path) { return }
    $parent = Split-Path -Parent ([IO.Path]::GetFullPath($Path))
    if ($parent -and -not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

if (-not $PSBoundParameters.ContainsKey('Output') -or -not $Output) {
    if ($Profile -eq "portrait") {
        $varsDir = Split-Path -Parent $varsFull
        $reelRoot = if ($varsDir) { Join-Path $varsDir "portrait_reels" } else { Join-Path (Split-Path -Parent $inputFull) "portrait_reels" }
        $cleanDir = Join-Path $reelRoot "clean"
        $baseName = Get-PortraitBaseName -InputPath $inputFull -VarsPath $varsFull
        Ensure-ParentDirectory (Join-Path $cleanDir "placeholder")
        $Output = Join-Path $cleanDir ($baseName + "_portrait_FINAL.mp4")
    } else {
        $Output = "autoframe_crop.mp4"
    }
}

Ensure-ParentDirectory $Output
Ensure-ParentDirectory $Debug
Ensure-ParentDirectory $Compare

. $Vars

if (-not ($cxExpr) -or -not ($cyExpr) -or -not ($zExpr)) {
    throw "Vars file must define `$cxExpr, `$cyExpr, and `$zExpr"
}

switch ($Profile) {
    "portrait" {
        $ratioW = 9.0
        $ratioH = 16.0
        $scale = "scale=1080:1920:flags=lanczos"
    }
    "landscape" {
        $ratioW = 16.0
        $ratioH = 9.0
        $scale = "scale=1920:1080:flags=lanczos"
    }
}

$cwExpr = "((ih*$ratioW/$ratioH)/$zExpr)"
$chExpr = "(ih/$zExpr)"
$wExpr = "floor(($cwExpr)/2)*2"
$hExpr = "floor(($chExpr)/2)*2"
$xCore = "($cxExpr)-($wExpr)/2"
$yCore = "($cyExpr)-($hExpr)/2"

$vf = @"
crop=w=$wExpr:h=$hExpr:
     x='clip($xCore,0,iw-($wExpr))':
     y='clip($yCore,0,ih-($hExpr))',
$scale,setsar=1,format=yuv420p
"@

Write-Host "[autoframe] Cropping with profile $Profile"

$crfStr = $Crf.ToString("0.###", [Globalization.CultureInfo]::InvariantCulture)
$ffmpegArgs = @(
    "-y",
    "-i", $Input,
    "-vf", $vf,
    "-c:v", "libx264",
    "-crf", $crfStr,
    "-preset", $Preset,
    "-c:a", "copy",
    $Output
)
& ffmpeg @ffmpegArgs

function Get-Fps([string]$Path) {
    $raw = & ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 $Path 2>$null
    if (-not $raw) { return 30.0 }
    $raw = $raw.Trim()
    $culture = [Globalization.CultureInfo]::InvariantCulture
    if ($raw -match '/') {
        $parts = $raw.Split('/')
        if ($parts.Count -eq 2) {
            [double]$num = 0
            [double]$den = 0
            if ([double]::TryParse($parts[0], [Globalization.NumberStyles]::Float, $culture, [ref]$num) -and
                [double]::TryParse($parts[1], [Globalization.NumberStyles]::Float, $culture, [ref]$den) -and
                $den -ne 0) {
                return $num / $den
            }
        }
    }
    try {
        return [double]::Parse($raw, $culture)
    } catch {
        return 30.0
    }
}

if ($Debug) {
    $fps = Get-Fps $Input
    if ($fps -le 0) { $fps = 30.0 }
    $fpsStr = $fps.ToString("0.#####", [Globalization.CultureInfo]::InvariantCulture)
    $wT = $wExpr -replace 'n',"(t*$fpsStr)"
    $hT = $hExpr -replace 'n',"(t*$fpsStr)"
    $xT = $xCore -replace 'n',"(t*$fpsStr)"
    $yT = $yCore -replace 'n',"(t*$fpsStr)"

    $debug_vf = @"
format=yuv420p,
drawbox:x='clip($xT,0,iw-($wT))':y='clip($yT,0,ih-($hT))':w=$wT:h=$hT:t=3:color=red@0.45
"@

    $dbgCrfStr = $DebugCrf.ToString("0.###", [Globalization.CultureInfo]::InvariantCulture)
    $dbgArgs = @(
        "-y",
        "-i", $Input,
        "-vf", $debug_vf,
        "-an",
        "-c:v", "libx264",
        "-crf", $dbgCrfStr,
        "-preset", $DebugPreset,
        $Debug
    )
    & ffmpeg @dbgArgs
}

if ($Compare) {
    if (-not $Debug) { throw "--Compare requires --Debug output" }
    $cmpArgs = @(
        "-y",
        "-i", $Output,
        "-i", $Debug,
        "-filter_complex", "[0:v]setpts=PTS-STARTPTS[main];[1:v]setpts=PTS-STARTPTS[dbg];[main][dbg]hstack=inputs=2",
        "-an",
        "-c:v", "libx264",
        "-crf", ($CompareCrf.ToString("0.###", [Globalization.CultureInfo]::InvariantCulture)),
        "-preset", $ComparePreset,
        $Compare
    )
    & ffmpeg @cmpArgs
}
