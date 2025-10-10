param(
    [Parameter(Mandatory=$true)] [string]$Csv,
    [string]$OutFile = ".\\out\\opener\\team_montage.mp4",
    [double]$Duration = 6.0,
    [switch]$KeepIntermediates,
    [string]$TempDir,
    [string]$BGPath = "C:\\Users\\scott\\soccer-video\\brand\\tsc\\end_card_1080x1920.png",
    [string]$BadgeSolidPath = "C:\\Users\\scott\\soccer-video\\brand\\tsc\\badge_clean.png",
    [string]$BadgeHolePath = "C:\\Users\\scott\\soccer-video\\brand\\tsc\\badge_hole.png"
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

if (-not (Test-Path $Csv)) {
    throw "Roster CSV not found: $Csv"
}

$entries = Import-Csv -LiteralPath $Csv
if (-not $entries -or $entries.Count -eq 0) {
    throw "Roster CSV is empty: $Csv"
}

$makeOpener = Join-Path $PSScriptRoot 'make_opener.ps1'
if (-not (Test-Path $makeOpener)) {
    throw "Dependency missing: $make_opener.ps1"
}

if (-not $TempDir) {
    $OutDir = Split-Path -Parent ([System.IO.Path]::GetFullPath($OutFile))
    if (-not $OutDir) { $OutDir = (Get-Location).Path }
    $TempDir = Join-Path $OutDir '__montage_tmp'
}

Ensure-ParentDirectory -Path $OutFile
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

$segments = @()
$index = 0
foreach ($entry in $entries) {
    $index += 1
    $name = $entry.name
    if (-not $name) { $name = $entry.PlayerName }
    $numberRaw = if ($entry.number) { $entry.number } elseif ($entry.PlayerNumber) { $entry.PlayerNumber } else { $null }
    $photo = if ($entry.photo) { $entry.photo } elseif ($entry.PlayerPhoto) { $entry.PlayerPhoto } else { $null }

    if (-not $name -or -not $numberRaw -or -not $photo) {
        throw "Row $index must include name, number, and photo columns."
    }

    if (-not (Test-Path $photo)) {
        throw "Player photo not found: $photo"
    }

    $number = 0
    if (-not [int]::TryParse(($numberRaw -replace '[^0-9]', ''), [ref]$number)) {
        throw "Player number must be numeric (row $index): $numberRaw"
    }

    $safeName = ($name -replace '[^\w\- ]','').Trim()
    if (-not $safeName) { $safeName = "Player$index" }
    $safeName = ($safeName -replace '\s+', '_')

    $segmentPath = Join-Path $TempDir ("{0:00}_{1}.mp4" -f $index, $safeName)

    $openArgs = @{
        PlayerName = $name
        PlayerNumber = $number
        PlayerPhoto = $photo
        OutDir = $TempDir
        OutFile = $segmentPath
        DUR = $Duration
        BGPath = $BGPath
        BadgeSolidPath = $BadgeSolidPath
        BadgeHolePath = $BadgeHolePath
    }

    Write-Host "Rendering opener segment $index for $name (#$number)"
    & $makeOpener @openArgs
    if ($LASTEXITCODE -ne 0) {
        throw "make_opener.ps1 failed for $name (#$number)"
    }

    if (-not (Test-Path $segmentPath)) {
        throw "Expected segment missing: $segmentPath"
    }

    $segments += $segmentPath
}

if ($segments.Count -eq 1) {
    Copy-Item -LiteralPath $segments[0] -Destination $OutFile -Force
} else {
    $concatPath = Join-Path $TempDir 'concat.txt'
    $segments | ForEach-Object {
        $escaped = $_.Replace("'", "''")
        "file '$escaped'"
    } | Set-Content -LiteralPath $concatPath -Encoding UTF8

    $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if (-not $ffmpegCmd) {
        throw "ffmpeg not found on PATH"
    }

    $ffArgs = @('-y','-f','concat','-safe','0','-i',$concatPath,'-c:v','libx264','-preset','slow','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k',$OutFile)
    Write-Host "Concatenating ${($segments.Count)} opener segments into montage"
    & ffmpeg @ffArgs
    if ($LASTEXITCODE -ne 0) {
        throw "ffmpeg concat failed with exit code $LASTEXITCODE"
    }
}

if (-not $KeepIntermediates) {
    Remove-Item -LiteralPath $TempDir -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "Montage ready at $OutFile"
