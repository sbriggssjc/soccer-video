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
    [switch]$CleanTemp,
    [switch]$Legacy
)

$ErrorActionPreference = 'Stop'

function Resolve-FullPath {
    param([string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

$clipPath = (Resolve-Path -LiteralPath $In).Path
$outPath = $null
if ($Out) {
    $outPath = Resolve-FullPath $Out
}

$portraitValue = $null
if ($PortraitSize) {
    $portraitValue = $PortraitSize
} elseif ($Portrait.IsPresent) {
    $portraitValue = '1080x1920'
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Split-Path -Parent $scriptRoot
$pythonCmd = Get-Command python -ErrorAction Stop
$unifiedScript = Join-Path $scriptRoot 'render_follow_unified.py'
$unifiedArgs = @($unifiedScript, '--in', $clipPath, '--src', $clipPath)
if ($outPath) { $unifiedArgs += @('--out', $outPath) }
if ($Preset) { $unifiedArgs += @('--preset', $Preset.ToLowerInvariant()) }
if ($portraitValue) { $unifiedArgs += @('--portrait', $portraitValue) }
if ($PSBoundParameters.ContainsKey('Fps')) { $unifiedArgs += @('--fps', [string]$Fps) }
if ($Flip180) { $unifiedArgs += '--flip180' }
if ($CleanTemp) { $unifiedArgs += '--clean-temp' }
if ($ExtraArgs) { $unifiedArgs += $ExtraArgs }

if (-not $Legacy.IsPresent) {
    Write-Host ('Invoking unified renderer: {0} {1}' -f $pythonCmd.Path, ($unifiedArgs -join ' '))
    & $pythonCmd.Path @unifiedArgs
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        exit 0
    }
    Write-Warning "Unified renderer failed (exit code $exitCode); attempting legacy fallback."
} else {
    Write-Host 'Legacy mode requested; skipping unified renderer.'
}

$presetKey = $Preset.ToLowerInvariant()
$legacyMap = @{
    'cinematic' = Join-Path $scriptRoot 'render_follow_autoz_cinematic_patched.py'
    'gentle'    = Join-Path $repoRoot 'render_follow_autoz_gentle.py'
    'realzoom'  = Join-Path $repoRoot 'render_follow_autoz_realzoom.py'
}
$legacyScript = $legacyMap[$presetKey]
if (-not $legacyScript -or -not (Test-Path $legacyScript)) {
    throw "Legacy script for preset '$Preset' not found."
}

$legacyArgs = @($legacyScript, '--in', $clipPath, '--src', $clipPath)
if ($outPath) { $legacyArgs += @('--out', $outPath) }
if ($portraitValue -and $presetKey -eq 'cinematic') {
    $parts = $portraitValue -split 'x'
    if ($parts.Length -eq 2) {
        $legacyArgs += @('--width', $parts[0], '--height', $parts[1])
    }
}
if ($CleanTemp -and $presetKey -eq 'cinematic') { $legacyArgs += '--clean-temp' }
if ($ExtraArgs) { $legacyArgs += $ExtraArgs }

Write-Host ('Invoking legacy renderer: {0} {1}' -f $pythonCmd.Path, ($legacyArgs -join ' '))
& $pythonCmd.Path @legacyArgs
exit $LASTEXITCODE
