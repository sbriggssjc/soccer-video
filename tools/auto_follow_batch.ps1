[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string[]]$Clips,
    [string]$OutDir,
    [switch]$ListOnly
)

$ErrorActionPreference = 'Stop'

function Resolve-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
    $resolved = Resolve-Path -LiteralPath $Path -ErrorAction SilentlyContinue
    if ($resolved) { return $resolved.Path }
    $combined = Join-Path (Get-Location) $Path
    $resolved = Resolve-Path -LiteralPath $combined -ErrorAction SilentlyContinue
    if ($resolved) { return $resolved.Path }
    return $Path
}

function Get-PythonCommand {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Path }
    $cmd = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Path }
    throw 'Python executable not found on PATH. Install Python 3.10+.'
}

$script:ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$script:RepoRoot = Split-Path -Parent $script:ScriptRoot
if (-not $PSBoundParameters.ContainsKey('OutDir') -or [string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $script:RepoRoot 'out'
}
$script:OutDir = Resolve-FullPath -Path $OutDir
$script:BallPlanner = Join-Path $script:RepoRoot 'tools\ball_path_offline.py'
$script:PythonExe = Get-PythonCommand

function Find-PlanJsonl {
    param([Parameter(Mandatory = $true)][string]$Stem)

    $diagRoot = Join-Path $script:OutDir 'follow_diag'
    $stemDir = Join-Path $diagRoot $Stem
    if (-not (Test-Path -LiteralPath $stemDir)) { return $null }

    $existing = Get-ChildItem -LiteralPath $stemDir -Recurse -File -Filter '*.jsonl' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($existing) { return $existing.FullName }
    return $null
}

function Ensure-PlanJsonl {
  param([string]$Stem, [string]$ClipPath)

  $existing = Find-PlanJsonl -Stem $Stem
  if ($existing) { return $existing }

  # Default plan path using your naming convention
  $plan = Join-Path $script:OutDir ("follow_diag\{0}\auto_hill\plan_dx-48_dy-108_sh-2.jsonl" -f $Stem)
  New-Item -Force -ItemType Directory -Path (Split-Path $plan) | Out-Null

  try {
    & $script:PythonExe $script:BallPlanner `
      --in "$ClipPath" `
      --out "$plan" `
      --dx -48 --dy -108 --sh 2
    if (Test-Path $plan) { return $plan }
  } catch {
    Write-Warning ("[PLAN] Auto-generation failed for {0}: {1}" -f $Stem, $_.Exception.Message)
  }

  return $null
}

$clipList = @()
foreach ($item in $Clips) {
    $resolved = Resolve-FullPath -Path $item
    if (-not (Test-Path -LiteralPath $resolved)) {
        Write-Warning ("[SKIP] Clip not found: {0}" -f $item)
        continue
    }
    $clipList += $resolved
}

if ($clipList.Count -eq 0) {
    Write-Warning 'No valid clips provided.'
    return
}

foreach ($Clip in $clipList) {
    $Stem = [System.IO.Path]::GetFileNameWithoutExtension($Clip)

    $Plan = Ensure-PlanJsonl -Stem $Stem -ClipPath $Clip
    if (-not $Plan) { Write-Host "[SKIP] No plan for $Stem (could not auto-generate)"; continue }

    if ($ListOnly) {
        Write-Host ("[LIST] {0} -> {1}" -f $Clip, $Plan)
        continue
    }

    Write-Host ("[READY] {0} using plan {1}" -f $Clip, $Plan)
}
