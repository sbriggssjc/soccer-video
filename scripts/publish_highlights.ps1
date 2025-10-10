param(
    [Parameter(Mandatory=$true)] [string]$Video,
    [string]$OutDir = ".\\out",
    [string]$Title = "",
    [string]$Subtitle = "",
    [string]$PublishDir,
    [switch]$Reencode,
    [switch]$KeepIntermediates
)

$ErrorActionPreference = 'Stop'

function Resolve-FullPath {
    param([string]$Path)
    if (-not $Path) { return $null }
    return ([System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path))
}

function Ensure-Directory {
    param([string]$Path)
    if (-not $Path) { return }
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

if (-not (Test-Path $Video)) {
    throw "Input video not found: $Video"
}

$Video = Resolve-FullPath -Path $Video
Ensure-Directory -Path $OutDir
$OutDir = ([System.IO.Path]::GetFullPath($OutDir))

$runScript = Join-Path $PSScriptRoot 'run_highlights.ps1'
if (-not (Test-Path $runScript)) {
    throw "Missing dependency: $runScript"
}

$runArgs = @{ Video = $Video; OutDir = $OutDir }
if ($Reencode) { $runArgs['Reencode'] = $true }

Write-Host "[1/3] Rebuilding highlight reels via run_highlights.ps1"
& $runScript @runArgs
if ($LASTEXITCODE -ne 0) {
    throw "run_highlights.ps1 failed with exit code $LASTEXITCODE"
}

$finalGoalsFirst = Join-Path $OutDir 'top_highlights_goals_first.mp4'
if (-not (Test-Path $finalGoalsFirst)) {
    throw "Expected highlight file missing: $finalGoalsFirst"
}

$brandScript = Join-Path $PSScriptRoot '..\\tools\\tsc_brand.ps1'
if (-not (Test-Path $brandScript)) {
    throw "Branding helper missing: $brandScript"
}

$brandedTemp = Join-Path $OutDir 'top_highlights_goals_first.branded.mp4'

$brandArgs = @{
    In = $finalGoalsFirst
    Out = $brandedTemp
    Aspect = '16x9'
    Watermark = $true
    EndCard = $true
}
if ($Title) { $brandArgs['Title'] = $Title }
if ($Subtitle) { $brandArgs['Subtitle'] = $Subtitle }

Write-Host "[2/3] Applying Tulsa SC branding"
& $brandScript @brandArgs
if ($LASTEXITCODE -ne 0) {
    throw "Branding step failed with exit code $LASTEXITCODE"
}

Move-Item -Force -LiteralPath $brandedTemp -Destination $finalGoalsFirst

if ($PublishDir) {
    $PublishDir = [System.IO.Path]::GetFullPath($PublishDir)
    Ensure-Directory -Path $PublishDir
    $destName = Join-Path $PublishDir ([System.IO.Path]::GetFileName($finalGoalsFirst))
    Copy-Item -LiteralPath $finalGoalsFirst -Destination $destName -Force
    Write-Host "[3/3] Copied branded highlight to $destName"
} else {
    Write-Host "[3/3] Branded highlight ready at $finalGoalsFirst"
}

if (-not $KeepIntermediates) {
    $logs = Join-Path $OutDir 'logs'
    if (Test-Path $logs) {
        Get-ChildItem -Path $logs -Filter '*.log' -Recurse | Where-Object { $_.Length -eq 0 } | Remove-Item -Force -ErrorAction SilentlyContinue
    }
}
