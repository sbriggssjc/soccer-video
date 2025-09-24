[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$In,
  [Parameter(Mandatory = $true)][string]$Vars,
  [Parameter(Mandatory = $true)][string]$Out,
  [Parameter(Mandatory = $true)][string]$VF
)

$ErrorActionPreference = 'Stop'

function Resolve-ExistingPath {
  param(
    [Parameter(Mandatory = $true)][string]$Path
  )

  try {
    return (Resolve-Path -LiteralPath $Path).ProviderPath
  }
  catch {
    throw "Path not found: $Path"
  }
}

function Resolve-TargetPath {
  param(
    [Parameter(Mandatory = $true)][string]$Path
  )

  return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Path))
}

Import-Module "$PSScriptRoot/Autoframe.psm1" -Force

$inPath   = Resolve-ExistingPath -Path $In
$varsPath = Resolve-ExistingPath -Path $Vars
$vfPath   = Resolve-TargetPath -Path $VF
$outPath  = Resolve-TargetPath -Path $Out

foreach ($dir in @($vfPath, $outPath) | ForEach-Object { Split-Path -Parent $_ } | Sort-Object -Unique) {
  if ($dir -and -not (Test-Path -LiteralPath $dir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
  }
}

Ensure-AutoframeTools

$exprs = Load-ExprVars -VarsPath $varsPath
$E     = Use-FFExprVars -InPath $inPath -Vars $exprs

Write-Host "--- DEBUG (computed) ---"
Write-Host "Input=$inPath"
Write-Host "Vars=$varsPath"
Write-Host "FPS=$($E.FPS)"
Write-Host "W=$($E.W)"
Write-Host "H=$($E.H)"
Write-Host "X=$($E.X)"
Write-Host "Y=$($E.Y)"

Write-Host "--- VF file path ---"
Write-Host $vfPath

$vfText = Write-CompareVF -VfPath $vfPath -E $E

Write-Host "--- VF file ---"
Write-Host $vfText

Render-Compare -In $inPath -VfPath $vfPath -Out $outPath
Write-Host "Done: $outPath"
