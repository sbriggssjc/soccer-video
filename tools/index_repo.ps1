param(
    [Parameter(Mandatory = $true)]
    [string]$Root,

    [switch]$IncludeOutputs,

    [string]$Globs = "**/*",

    [string]$ExcludeGlobs = "",

    [double]$MaxHashMB = 25,

    [int]$Workers = [Environment]::ProcessorCount
)

function Write-Info {
    param([string]$Message)
    Write-Host "[index_repo] $Message"
}

if (-not (Test-Path -LiteralPath $Root)) {
    throw "Root path does not exist: $Root"
}

$rootPath = (Resolve-Path -LiteralPath $Root).Path
$outDir = Join-Path -Path $rootPath -ChildPath "out"

if (-not (Test-Path -LiteralPath $outDir)) {
    Write-Info "Creating output directory at $outDir"
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py"
}

if (-not $pythonCmd) {
    Write-Warning "Python executable not found on PATH. Please install Python 3.10+ and retry."
    exit 1
}

$scriptPath = Join-Path -Path $rootPath -ChildPath "tools/index_repo.py"
if (-not (Test-Path -LiteralPath $scriptPath)) {
    throw "Python script not found at $scriptPath"
}

$argsList = @("--root", $rootPath, "--globs", $Globs, "--exclude-globs", $ExcludeGlobs, "--max-hash-mb", $MaxHashMB.ToString(), "--workers", $Workers.ToString())
if ($IncludeOutputs) {
    $argsList += "--include-outputs"
}

Write-Info "Running indexer with Python at $pythonCmd"
Write-Info "Arguments: $($argsList -join ' ')"

$processInfo = New-Object System.Diagnostics.ProcessStartInfo
$processInfo.FileName = $pythonCmd
$processInfo.ArgumentList.Add($scriptPath)
foreach ($arg in $argsList) {
    $processInfo.ArgumentList.Add($arg)
}
$processInfo.RedirectStandardOutput = $true
$processInfo.RedirectStandardError = $true
$processInfo.UseShellExecute = $false
$processInfo.CreateNoWindow = $true

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $processInfo
$null = $proc.Start()

while (-not $proc.HasExited) {
    if (-not $proc.StandardOutput.EndOfStream) {
        Write-Host $proc.StandardOutput.ReadLine()
    }
    Start-Sleep -Milliseconds 100
}

while (-not $proc.StandardOutput.EndOfStream) {
    Write-Host $proc.StandardOutput.ReadLine()
}

if (-not $proc.StandardError.EndOfStream) {
    Write-Warning $proc.StandardError.ReadToEnd()
}

if ($proc.ExitCode -ne 0) {
    throw "index_repo.py exited with code $($proc.ExitCode)"
}

$artifacts = @(
    "repo_index.json",
    "repo_index.csv",
    "repo_index.xlsx",
    "folder_rollup.csv",
    "duplicates.csv",
    "process_graph.dot",
    "process_graph.png",
    "orphan_signals.csv",
    "SUMMARY.md",
    "index_repo.log"
)

Write-Info "Artifacts generated:"
foreach ($artifact in $artifacts) {
    $path = Join-Path -Path $outDir -ChildPath $artifact
    if (Test-Path -LiteralPath $path) {
        Write-Info " - $path"
    }
}

