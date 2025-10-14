[CmdletBinding()]
param(
    [switch]$WhatIf,
    [int]$AudioCrossfadeMs = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$OpenerPath = 'C:\Users\scott\soccer-video\out\opener\opener_v1.mp4'
$SourceRoot = 'C:\Users\scott\soccer-video\out\postables\in'
$OutputRoot = 'C:\Users\scott\soccer-video\out\postables\with_opener'
$TempRoot = 'C:\Users\scott\soccer-video\out\postables\_tmp_conformed'

$SupportedExtensions = @('*.mp4', '*.mov', '*.m4v')
$TargetFrameWidth = 1080
$TargetFrameHeight = 1920
$TargetFrameRate = 30
$TargetAudioSampleRate = 48000
$TargetAudioChannels = 2
$TargetVideoCodec = 'h264'
$TargetPixelFormat = 'yuv420p'
$TargetAudioCodec = 'aac'
$DefaultCRF = 20

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO ] $Message"
}

function Write-Warn {
    param([string]$Message)
    Write-Warning $Message
}

function Write-ErrorMessage {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Get-RelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )
    $base = [System.IO.Path]::GetFullPath($BasePath)
    if (-not $base.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $base += [System.IO.Path]::DirectorySeparatorChar
    }
    $baseUri = [Uri]::new($base)
    $targetUri = [Uri]::new([System.IO.Path]::GetFullPath($TargetPath))
    $relative = $baseUri.MakeRelativeUri($targetUri).ToString()
    return [Uri]::UnescapeDataString($relative).Replace('/', [System.IO.Path]::DirectorySeparatorChar)
}

function Get-ConcatFileLine {
    param([Parameter(Mandatory = $true)][string]$Path)
    $escaped = $Path -replace "'", "'\\''"
    return "file '$escaped'"
}

function Invoke-ExternalCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [switch]$SkipExecution
    )
    $renderedArgs = $Arguments | ForEach-Object {
        if ($_ -match '^[A-Za-z0-9_.\\/:-]+$') { $_ } else { '"' + ($_ -replace '"', '""') + '"' }
    }
    Write-Info "Running: $FilePath $($renderedArgs -join ' ')"
    if ($SkipExecution) {
        return 0
    }
    & $FilePath @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Command '$FilePath' exited with code $exitCode"
    }
    return $exitCode
}

function Get-MediaInfo {
    param([Parameter(Mandatory = $true)][string]$Path)
    $ffprobeArgs = @('-v', 'error', '-show_streams', '-show_format', '-of', 'json', $Path)
    $json = & ffprobe @ffprobeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "ffprobe failed for '$Path'"
    }
    $data = $json | ConvertFrom-Json
    $videoStream = $data.streams | Where-Object { $_.codec_type -eq 'video' } | Select-Object -First 1
    $audioStream = $data.streams | Where-Object { $_.codec_type -eq 'audio' } | Select-Object -First 1
    $duration = $null
    if ($data.format -and $data.format.duration) {
        [double]::TryParse($data.format.duration, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$duration) | Out-Null
    }
    $fps = $null
    if ($videoStream.avg_frame_rate -and $videoStream.avg_frame_rate -ne '0/0') {
        $parts = $videoStream.avg_frame_rate -split '/'
        if ($parts.Length -eq 2) {
            $numerator = [double]::Parse($parts[0], [System.Globalization.CultureInfo]::InvariantCulture)
            $denominator = [double]::Parse($parts[1], [System.Globalization.CultureInfo]::InvariantCulture)
            if ($denominator -ne 0) {
                $fps = $numerator / $denominator
            }
        }
    }
    return [pscustomobject]@{
        Path       = $Path
        Video      = $videoStream
        Audio      = $audioStream
        Duration   = $duration
        FrameRate  = $fps
    }
}

function Test-MatchesBrandSpec {
    param([Parameter(Mandatory = $true)]$MediaInfo)
    if (-not $MediaInfo.Video -or -not $MediaInfo.Audio) {
        return $false
    }
    $video = $MediaInfo.Video
    $audio = $MediaInfo.Audio
    if ($video.width -ne $TargetFrameWidth -or $video.height -ne $TargetFrameHeight) {
        return $false
    }
    if ($MediaInfo.FrameRate) {
        if ([math]::Abs($MediaInfo.FrameRate - $TargetFrameRate) -gt 0.01) {
            return $false
        }
    } else {
        return $false
    }
    if (($video.codec_name -ne $TargetVideoCodec) -or ($video.pix_fmt -ne $TargetPixelFormat)) {
        return $false
    }
    if ($audio.codec_name -ne $TargetAudioCodec) {
        return $false
    }
    $sampleRate = 0
    if ($audio.sample_rate) {
        [int]::TryParse($audio.sample_rate, [ref]$sampleRate) | Out-Null
    }
    if ($sampleRate -ne $TargetAudioSampleRate) {
        return $false
    }
    if ($audio.channels -ne $TargetAudioChannels) {
        return $false
    }
    return $true
}

function Ensure-Directory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        if ($WhatIf) {
            Write-Info "Would create directory: $Path"
        } else {
            Write-Info "Creating directory: $Path"
            New-Item -ItemType Directory -Path $Path -Force | Out-Null
        }
    }
}

if ($AudioCrossfadeMs -lt 0) {
    throw "AudioCrossfadeMs must be zero or positive."
}

Get-Command ffmpeg -ErrorAction Stop | Out-Null
Get-Command ffprobe -ErrorAction Stop | Out-Null

if (-not (Test-Path -LiteralPath $OpenerPath)) {
    throw "Opener file not found: $OpenerPath"
}

Ensure-Directory -Path $OutputRoot
Ensure-Directory -Path $TempRoot

$candidateFiles = @()
foreach ($pattern in $SupportedExtensions) {
    $candidateFiles += Get-ChildItem -Path $SourceRoot -Recurse -File -Include $pattern
}
$candidateFiles = $candidateFiles | Sort-Object FullName -Unique

$targetList = @()
foreach ($file in $candidateFiles) {
    $relative = Get-RelativePath -BasePath $SourceRoot -TargetPath $file.FullName
    $targetList += $relative
}

Write-Host 'TARGET REELS'
if ($targetList.Count -eq 0) {
    Write-Host '  (none found)'
} else {
    foreach ($item in $targetList) {
        Write-Host "  $item"
    }
}
Write-Host ''

if ($WhatIf) {
    Write-Info 'Dry run requested (-WhatIf). No changes will be made.'
}

$openerInfo = Get-MediaInfo -Path $OpenerPath
$openerDuration = if ($openerInfo.Duration) { [math]::Round($openerInfo.Duration, 3) } else { $null }

$summary = New-Object System.Collections.Generic.List[object]
$failures = New-Object System.Collections.Generic.List[object]

foreach ($sourceFile in $candidateFiles) {
    $relativePath = Get-RelativePath -BasePath $SourceRoot -TargetPath $sourceFile.FullName
    $outputFileName = ([System.IO.Path]::GetFileNameWithoutExtension($sourceFile.Name)) + '_with_opener.mp4'
    $outputPath = Join-Path $OutputRoot $outputFileName

    Write-Info "Processing '$relativePath'"

    try {
        $sourceInfo = Get-MediaInfo -Path $sourceFile.FullName
        $needsConform = -not (Test-MatchesBrandSpec -MediaInfo $sourceInfo)
        $videoSourcePath = $sourceFile.FullName
        $conformedPath = $null
        $conformedPerformed = $false

        $safeName = ($relativePath -replace "[:\\/ '\"\[\]]", '_')
        if (-not $safeName) { $safeName = [Guid]::NewGuid().ToString('N') }

        if ($needsConform) {
            $conformedPath = Join-Path $TempRoot ($safeName + '_conformed.mp4')
            $ffmpegArgs = @(
                '-y',
                '-i', $sourceFile.FullName,
                '-vf', "scale=$TargetFrameWidth:$TargetFrameHeight:force_original_aspect_ratio=decrease,pad=$TargetFrameWidth:$TargetFrameHeight:($TargetFrameWidth-iw)/2:($TargetFrameHeight-ih)/2:black",
                '-r', $TargetFrameRate,
                '-c:v', 'libx264',
                '-crf', $DefaultCRF,
                '-pix_fmt', $TargetPixelFormat,
                '-c:a', 'aac',
                '-ar', $TargetAudioSampleRate,
                '-ac', $TargetAudioChannels,
                $conformedPath
            )
            if ($WhatIf) {
                Write-Info "Would conform '$relativePath' to brand spec -> $conformedPath"
            } else {
                Invoke-ExternalCommand -FilePath 'ffmpeg' -Arguments $ffmpegArgs
            }
            $videoSourcePath = $conformedPath
            $conformedPerformed = $true
            if (-not $WhatIf) {
                $sourceInfo = Get-MediaInfo -Path $videoSourcePath
            }
        }

        $concatListPath = Join-Path $TempRoot ($safeName + '_concat.txt')
        $concatLines = @(
            Get-ConcatFileLine -Path $OpenerPath,
            Get-ConcatFileLine -Path $videoSourcePath
        )
        if ($WhatIf) {
            Write-Info "Would create concat list at $concatListPath"
        } else {
            Set-Content -Path $concatListPath -Value $concatLines -Encoding UTF8
        }

        $clipDuration = if ($sourceInfo.Duration) { [math]::Round($sourceInfo.Duration, 3) } else { $null }

        if ($AudioCrossfadeMs -gt 0) {
            $videoOnlyPath = Join-Path $TempRoot ($safeName + '_video.mp4')
            $audioTempPath = Join-Path $TempRoot ($safeName + '_audio.m4a')
            $crossfadeSeconds = [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, '{0:F3}', $AudioCrossfadeMs / 1000)

            $videoArgs = @('-y', '-f', 'concat', '-safe', '0', '-i', $concatListPath, '-map', '0:v', '-c:v', 'copy', '-an', $videoOnlyPath)
            if ($WhatIf) {
                Write-Info "Would concat video streams -> $videoOnlyPath"
            } else {
                Invoke-ExternalCommand -FilePath 'ffmpeg' -Arguments $videoArgs
            }

            $audioSourceForCrossfade = $videoSourcePath
            $audioArgs = @(
                '-y',
                '-i', $OpenerPath,
                '-i', $audioSourceForCrossfade,
                '-filter_complex', "[0:a]aresample=$TargetAudioSampleRate,asetpts=PTS-STARTPTS[a0];[1:a]aresample=$TargetAudioSampleRate,asetpts=PTS-STARTPTS[a1];[a0][a1]acrossfade=d=$crossfadeSeconds:c1=tri:c2=tri[aout]",
                '-map', '[aout]',
                '-c:a', 'aac',
                '-ar', $TargetAudioSampleRate,
                '-ac', $TargetAudioChannels,
                $audioTempPath
            )
            if ($WhatIf) {
                Write-Info "Would create audio crossfade ($AudioCrossfadeMs ms) -> $audioTempPath"
            } else {
                Invoke-ExternalCommand -FilePath 'ffmpeg' -Arguments $audioArgs
            }

            $muxArgs = @('-y', '-i', $videoOnlyPath, '-i', $audioTempPath, '-c:v', 'copy', '-c:a', 'copy', $outputPath)
            if ($WhatIf) {
                Write-Info "Would mux video + audio into $outputPath"
            } else {
                Invoke-ExternalCommand -FilePath 'ffmpeg' -Arguments $muxArgs
            }
        }
        else {
            $concatArgs = @('-y', '-f', 'concat', '-safe', '0', '-i', $concatListPath, '-c', 'copy', $outputPath)
            if ($WhatIf) {
                Write-Info "Would concat opener + clip via stream copy -> $outputPath"
            } else {
                Invoke-ExternalCommand -FilePath 'ffmpeg' -Arguments $concatArgs
            }
        }

        $totalDuration = $null
        if (-not $WhatIf) {
            $outputInfo = Get-MediaInfo -Path $outputPath
            if ($outputInfo.Duration) {
                $totalDuration = [math]::Round($outputInfo.Duration, 3)
            }
        }

        $summary.Add([pscustomobject]@{
            Source        = $relativePath
            Conformed     = $(if ($conformedPerformed) { 'Yes' } else { 'No' })
            Output        = $outputPath
            OpenerSeconds = $openerDuration
            ClipSeconds   = $clipDuration
            TotalSeconds  = $totalDuration
            Status        = 'OK'
        }) | Out-Null
    }
    catch {
        $message = $_.Exception.Message
        Write-ErrorMessage "Failed '$relativePath': $message"
        $summary.Add([pscustomobject]@{
            Source        = $relativePath
            Conformed     = 'Failed'
            Output        = $outputPath
            OpenerSeconds = $openerDuration
            ClipSeconds   = $null
            TotalSeconds  = $null
            Status        = 'FAILED'
        }) | Out-Null
        $failures.Add([pscustomobject]@{ Source = $relativePath; Reason = $message }) | Out-Null
        continue
    }
}

if ($summary.Count -gt 0) {
    Write-Host ''
    Write-Host 'Processing Summary:'
    $summary | Format-Table -AutoSize | Out-String | Write-Host
}

if ($failures.Count -gt 0) {
    Write-ErrorMessage 'One or more files failed to process:'
    foreach ($failure in $failures) {
        Write-ErrorMessage "  $($failure.Source): $($failure.Reason)"
    }
    exit 1
}

Write-Info 'Completed successfully.'
exit 0
