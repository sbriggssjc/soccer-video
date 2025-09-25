<#
.SYNOPSIS
  Generate missing autoframe vars files for a fixed list of highlight clips.
.DESCRIPTION
  Locates the autoframe generator within the repository (PowerShell preferred,
  otherwise falls back to the Python tracker + fitter pipeline), removes any
  invalid stub vars, regenerates them at 24 fps, and validates that the
  resulting files contain cxExpr/cyExpr/zExpr assignments. Designed to be run
  from the repo root.
#>
[CmdletBinding()]
param(
  [string]$Root = (Split-Path -Parent (Resolve-Path $PSScriptRoot)),
  [string]$InputDir = $null,
  [string]$VarsDir  = $null,
  [int]$Fps = 24,
  [string[]]$Targets = @(
    '006__GOAL__t699.30-t711.50',
    '007__GOAL__t982.00-t987.90',
    '008__GOAL__t1247.80-t1259.20',
    '009__GOAL__t1420.40-t1438.20',
    '010__GOAL__t1541.10-t1547.90',
    '011__GOAL__t1655.00-t1670.30',
    '012__GOAL__t1705.20-t1708.50',
    '013__GOAL__t1767.50-t1772.60',
    '014__GOAL__t1904.20-t1925.20',
    '015__DRIBBLING__t1972.50-t1980.70',
    '016__SHOT__t2029.90-t2039.50',
    '017__SHOT__t2284.90-t2298.20',
    '018__SHOT__t2419.50-t2425.60',
    '019__SHOT__t2633.00-t2638.10',
    '020__GOAL__t2816.60-t2831.80',
    '021__SHOT__t2860.70-t2873.00',
    '022__SHOT__t3028.10-t3059.70',
    '023__SHOT__t3098.90-t3108.80'
  )
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $InputDir) {
  $InputDir = Join-Path $Root 'out\atomic_clips'
}
if (-not $VarsDir) {
  $VarsDir = Join-Path $Root 'out\autoframe_work'
}

function Invoke-ExternalCommand {
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter()][string[]]$Arguments = @(),
    [string]$DisplayName = $null
  )

  $cmdName = if ($DisplayName) { $DisplayName } else { $FilePath }
  Write-Host "Running: $cmdName $($Arguments -join ' ')" -ForegroundColor DarkGray
  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed (exit $LASTEXITCODE): $cmdName"
  }
}

function Test-VarsFile {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) { return $false }
  $content = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
  foreach ($token in 'cxExpr', 'cyExpr', 'zExpr') {
    if ($content -notmatch "(?im)^\s*\$$token\s*=") {
      return $false
    }
  }
  return $true
}

function Normalize-VarsFile {
  param([Parameter(Mandatory)][string]$Path)

  if (-not (Test-Path -LiteralPath $Path)) { return $false }
  $content = Get-Content -LiteralPath $Path -Raw -Encoding UTF8

  # Already OK?
  if ($content -match '(?im)^\s*\$cxExpr\s*=') { return $true }

  # Try to extract cx/cy/z in many shapes: cx:, cx =, cxExpr =, with/without "$"
  $cx = ($content | Select-String -Pattern '^\s*(\$?\s*cx(?:Expr)?\s*[:=])\s*(.+)$' -AllMatches).Matches |
        Select-Object -First 1 | ForEach-Object { $_.Groups[2].Value }
  $cy = ($content | Select-String -Pattern '^\s*(\$?\s*cy(?:Expr)?\s*[:=])\s*(.+)$' -AllMatches).Matches |
        Select-Object -First 1 | ForEach-Object { $_.Groups[2].Value }
  $z  = ($content | Select-String -Pattern '^\s*(\$?\s*z(?:Expr)?\s*[:=])\s*(.+)$'  -AllMatches).Matches |
        Select-Object -First 1 | ForEach-Object { $_.Groups[2].Value }

  if ($cx -and $cy -and $z) {
@"
`$cxExpr = $cx
`$cyExpr = $cy
`$zExpr  = $z
"@ | Set-Content -LiteralPath $Path -Encoding UTF8
    return $true
  }
  return $false
}

function Remove-IfInvalidVars {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) { return }
  if (-not (Test-VarsFile -Path $Path)) {
    Write-Host "Removing invalid vars stub: $Path" -ForegroundColor Yellow
    Remove-Item -LiteralPath $Path -Force
  }
}

function Find-PowerShellGenerator {
  param([string]$RootPath)

  Get-ChildItem -Path $RootPath -Recurse -File -Filter '*.ps1' |
    Where-Object {
      $_.FullName -ne $PSCommandPath -and
      $_.Name -notmatch '^run-compare-batch\.ps1$' -and
      (Select-String -LiteralPath $_.FullName -Pattern '(?im)^\s*param\(' -Quiet) -and
      (Select-String -LiteralPath $_.FullName -Pattern '(?i)\b-?Input\b' -Quiet) -and
      (Select-String -LiteralPath $_.FullName -Pattern '(?i)\bVarsOut\b' -Quiet)
    } |
    Select-Object -First 1
}

function Find-PythonTracker {
  param([string]$RootPath)
  $candidates = Get-ChildItem -Path $RootPath -Recurse -File -Filter '*.py' |
    Where-Object {
      (Select-String -LiteralPath $_.FullName -Pattern 'add_argument\("--csv"' -Quiet) -and
      (Select-String -LiteralPath $_.FullName -Pattern 'add_argument\("--in"' -Quiet)
    }
  if ($candidates) {
    return ($candidates | Sort-Object FullName | Select-Object -First 1)
  }
  return $null
}

function Find-PythonFitter {
  param([string]$RootPath)
  $candidates = Get-ChildItem -Path $RootPath -Recurse -File -Filter '*.py' |
    Where-Object {
      (Select-String -LiteralPath $_.FullName -Pattern 'write_ps1vars' -Quiet) -or
      (Select-String -LiteralPath $_.FullName -Pattern '\$cxExpr' -Quiet)
    }
  if ($candidates) {
    # prefer fit_expr.py specifically when present
    $fitExpr = $candidates | Where-Object { $_.Name -ieq 'fit_expr.py' } | Select-Object -First 1
    if ($fitExpr) { return $fitExpr }
    return ($candidates | Sort-Object FullName | Select-Object -First 1)
  }
  return $null
}

$psGenerator = Find-PowerShellGenerator -RootPath $Root

# ⬇️ Prevent self-selection
if ($psGenerator -and ($psGenerator.FullName -eq $PSCommandPath)) {
  $psGenerator = $null
}
$pythonTracker = $null
$pythonFitter = $null
$generatorMode = $null
$generatorDescription = $null

if ($psGenerator) {
  $generatorMode = 'PowerShell'
  $generatorDescription = $psGenerator.FullName
  Write-Host "Using PowerShell generator: $generatorDescription" -ForegroundColor Cyan
}
else {
  $pythonTracker = Find-PythonTracker -RootPath $Root
  $pythonFitter = Find-PythonFitter -RootPath $Root
  if ($pythonTracker -and $pythonFitter) {
    $generatorMode = 'PythonPipeline'
    $generatorDescription = "python $($pythonTracker.FullName) -> python $($pythonFitter.FullName)"
    Write-Host "Using Python autoframe pipeline:" -ForegroundColor Cyan
    Write-Host "  tracker: $($pythonTracker.FullName)" -ForegroundColor Cyan
    Write-Host "  fitter : $($pythonFitter.FullName)" -ForegroundColor Cyan
  }
}

if (-not $generatorMode) {
  Write-Host "Unable to locate an autoframe vars generator." -ForegroundColor Red
  Write-Host "Searched for PowerShell scripts containing 'VarsOut' and Python scripts containing 'write_ps1vars'." -ForegroundColor Red
  $hintCandidates = Get-ChildItem -Path $Root -Recurse -File -Include '*.ps1','*.py' |
    Where-Object { Select-String -LiteralPath $_.FullName -Pattern 'cxExpr' -Quiet }
  if ($hintCandidates) {
    Write-Host "Scripts mentioning 'cxExpr' for manual inspection:" -ForegroundColor Yellow
    $hintCandidates | Sort-Object FullName | Select-Object -First 10 | ForEach-Object {
      Write-Host "  $_"
    }
  }
  throw "Autoframe generator not found."
}

if (-not (Test-Path -LiteralPath $InputDir)) {
  throw "Input directory not found: $InputDir"
}
if (-not (Test-Path -LiteralPath $VarsDir)) {
  New-Item -ItemType Directory -Path $VarsDir -Force | Out-Null
}

$pythonExe = $env:PYTHON
if (-not $pythonExe) {
  $pythonExe = 'python'
}

$results = @()
foreach ($target in $Targets) {
  $inputPath = Join-Path $InputDir "$target.mp4"
  if (-not (Test-Path -LiteralPath $inputPath)) {
    throw "Missing input clip: $inputPath"
  }

  $varsPath = Join-Path $VarsDir "$target`_zoom.ps1vars"
  $csvPath  = Join-Path $VarsDir "$target`_zoom.csv"

  Remove-IfInvalidVars -Path $varsPath
  if (Test-Path -LiteralPath $csvPath) {
    Remove-Item -LiteralPath $csvPath -Force
  }

  switch ($generatorMode) {
    'PowerShell' {
      $args = @('-Input', $inputPath, '-VarsOut', $varsPath, '-Fps', $Fps, '-Verbose')
      Invoke-ExternalCommand -FilePath $psGenerator.FullName -Arguments $args -DisplayName $psGenerator.FullName
    }
    'PythonPipeline' {
      $trackerArgs = @($pythonTracker.FullName, '--in', $inputPath, '--csv', $csvPath, '--profile', 'portrait', '--roi', 'goal', '--goal_side', 'auto')
      Invoke-ExternalCommand -FilePath $pythonExe -Arguments $trackerArgs -DisplayName "$pythonExe $([IO.Path]::GetFileName($pythonTracker.FullName))"

      if (-not (Test-Path -LiteralPath $csvPath)) {
        throw "Tracker did not produce CSV: $csvPath"
      }

      $fitterArgs = @($pythonFitter.FullName, '--csv', $csvPath, '--out', $varsPath, '--profile', 'portrait')
      Invoke-ExternalCommand -FilePath $pythonExe -Arguments $fitterArgs -DisplayName "$pythonExe $([IO.Path]::GetFileName($pythonFitter.FullName))"
    }
    default {
      throw "Unsupported generator mode: $generatorMode"
    }
  }

  # If the strict check fails, try to normalize once
  if (-not (Test-VarsFile -Path $varsPath)) {
    if (Normalize-VarsFile -Path $varsPath -and (Test-VarsFile -Path $varsPath)) {
      Write-Host "Normalized vars format for $target" -ForegroundColor Yellow
    }
    else {
      throw "Generated vars missing cx/cy/z expressions for $target (generator: $generatorDescription)"
    }
  }

  $results += [pscustomobject]@{
    Target = $target
    VarsPath = $varsPath
  }
}

Write-Host ""  # spacing
Write-Host "Successfully generated vars for $($results.Count) clips." -ForegroundColor Green
Write-Host "Generator: $generatorDescription" -ForegroundColor Green
foreach ($entry in $results) {
  Write-Host "  $($entry.Target) -> $($entry.VarsPath)" -ForegroundColor Green
}
