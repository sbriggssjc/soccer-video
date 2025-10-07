param(
  [Parameter(Mandatory=$true)] [string]$Root,
  [switch]$IncludeOutputs,
  [string]$Globs = "**/*",
  [string]$ExcludeGlobs = "",
  [int]$Workers = [Environment]::ProcessorCount,
  [double]$MaxHashMB = 25
)

$ErrorActionPreference = "Stop"

# Resolve repo root & out dir
$RootPath = (Resolve-Path -LiteralPath $Root).ProviderPath
$outDir = Join-Path -Path $RootPath -ChildPath "out"
if (!(Test-Path -LiteralPath $outDir)) {
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

# Find python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
  Write-Host "Python not found in PATH. Please install Python 3.10+."
  exit 1
}

# Build arg list
$pyArgs = @(
  "tools/index_repo.py",
  "--root", "$RootPath",
  "--globs", "$Globs",
  "--exclude-globs", "$ExcludeGlobs",
  "--workers", "$Workers",
  "--max-hash-mb", "$MaxHashMB"
)
if ($IncludeOutputs) { $pyArgs += "--include-outputs" }

# Run
Write-Host "Running indexer with root: $RootPath"
& $python.Source $pyArgs

Write-Host ""
Write-Host "Artifacts:"
Get-ChildItem -Path $outDir -File | ForEach-Object { " - $($_.FullName)" }
