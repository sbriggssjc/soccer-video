param(
    [Parameter(Mandatory = $true)]
    [string]$In,
    [string]$Out,
    [ValidateSet('Cinematic','Gentle','RealZoom')]
    [string]$Preset = 'Cinematic',
    [switch]$Portrait,
    [string]$PortraitSize,
    [double]$Fps,
    [switch]$Flip180,
    [string[]]$ExtraArgs,
    [switch]$CleanTemp
)

$ErrorActionPreference = 'Stop'

function Resolve-FullPath {
    param([string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Get-PythonCommand {
    $candidate = Get-Command python -ErrorAction SilentlyContinue
    if ($candidate) { return $candidate }
    $candidate = Get-Command py -ErrorAction SilentlyContinue
    if ($candidate) { return $candidate }
    throw 'Python executable not found on PATH. Install Python 3.10+.'
}

function Get-PortraitValue {
    param(
        [string]$PresetName,
        [string]$RequestedSize,
        [string]$ScriptRootPath,
        [switch]$PortraitSwitch
    )

    if ($RequestedSize) { return $RequestedSize }
    if (-not $PortraitSwitch.IsPresent) { return $null }

    $presetsPath = Join-Path $ScriptRootPath 'render_presets.yaml'
    if (-not (Test-Path $presetsPath)) { return '1080x1920' }
    $lines = Get-Content -LiteralPath $presetsPath
    $current = ''
    foreach ($line in $lines) {
        if ($line -match '^([A-Za-z0-9_]+):\s*$') {
            $current = $Matches[1].ToLowerInvariant()
            continue
        }
        if ($current -ne $PresetName.ToLowerInvariant()) { continue }
        if ($line -match '^\s+portrait:\s*"?([0-9]+x[0-9]+)"?') {
            return $Matches[1]
        }
    }
    return '1080x1920'
}

$clipPath = (Resolve-Path -LiteralPath $In).Path
$outPath = $null
if ($Out) { $outPath = Resolve-FullPath $Out }


$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$portraitValue = Get-PortraitValue -PresetName $Preset -RequestedSize $PortraitSize -ScriptRootPath $scriptRoot -PortraitSwitch $Portrait
$pythonCmd = Get-PythonCommand
$unifiedScript = Join-Path $scriptRoot 'render_follow_unified.py'

$argsList = New-Object System.Collections.Generic.List[string]
$argsList.Add($unifiedScript)
$argsList.Add('--in')
$argsList.Add($clipPath)
$argsList.Add('--src')
$argsList.Add($clipPath)
$argsList.Add('--preset')
$argsList.Add($Preset.ToLowerInvariant())
if ($outPath) {
    $argsList.Add('--out')
    $argsList.Add($outPath)
}
if ($portraitValue) {
    $argsList.Add('--portrait')
    $argsList.Add($portraitValue)
}
if ($PSBoundParameters.ContainsKey('Fps')) {
    $argsList.Add('--fps')
    $argsList.Add([string]$Fps)
}
if ($Flip180.IsPresent) { $argsList.Add('--flip180') }
if ($CleanTemp.IsPresent) { $argsList.Add('--clean-temp') }
if ($ExtraArgs) {
    foreach ($item in $ExtraArgs) { $argsList.Add($item) }
}

$commandLine = $argsList -join ' '
Write-Host ('Invoking unified renderer: {0} {1}' -f $pythonCmd.Path, $commandLine)

& $pythonCmd.Path $argsList.ToArray()
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Error ("Unified renderer failed with exit code {0}" -f $exitCode)
}
exit $exitCode
