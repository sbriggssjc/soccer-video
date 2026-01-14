[CmdletBinding()]
param(
  [string]$RepoRoot = "C:\Users\scott\soccer-video",
  [string]$VaultRoot = "C:\Users\scott\OneDrive\SoccerVideoMedia",
  [switch]$Apply
)

$ErrorActionPreference = "Stop"
$LogDir = Join-Path $RepoRoot "out\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir "storage_optimize.log"

function Log($msg){
  $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $line = "$ts $msg"
  $line | Tee-Object -FilePath $LogPath -Append | Out-Host
}

function Is-JunctionOrSymlink([string]$path){
  if (-not (Test-Path -LiteralPath $path)) { return $false }
  $item = Get-Item -LiteralPath $path -Force
  return [bool]($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
}

function Ensure-Dir([string]$path){
  New-Item -ItemType Directory -Force -Path $path | Out-Null
}

function Folder-SizeGB([string]$path){
  if (-not (Test-Path -LiteralPath $path)) { return 0.0 }
  $sum = (Get-ChildItem -LiteralPath $path -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
  if (-not $sum) { $sum = 0 }
  return [Math]::Round($sum / 1GB, 2)
}

function Move-And-Junction([string]$src, [string]$dst){
  # src is inside repo, dst is inside vault
  if (-not (Test-Path -LiteralPath $src)) { Log "[skip] missing: $src"; return }

  if (Is-JunctionOrSymlink $src){
    Log "[skip] already reparsepoint (junction/symlink): $src"
    return
  }

  Ensure-Dir (Split-Path -Parent $dst)

  if (Test-Path -LiteralPath $dst){
    Log "[warn] destination exists: $dst"
  }

  if (-not $Apply){
    Log "[plan] move '$src' -> '$dst' and create junction at '$src'"
    return
  }

  Log "[do] moving '$src' -> '$dst'"
  Move-Item -LiteralPath $src -Destination $dst -Force

  # Create junction back
  Log "[do] mklink /J '$src' -> '$dst'"
  $cmd = "mklink /J `"$src`" `"$dst`""
  cmd /c $cmd | Out-Null

  if (-not (Is-JunctionOrSymlink $src)){
    throw "Junction creation failed for $src"
  }
}

Log "=== Optimize-OneDriveVault starting (Apply=$Apply) ==="
Log "RepoRoot : $RepoRoot"
Log "VaultRoot: $VaultRoot"

Ensure-Dir $VaultRoot
Ensure-Dir (Join-Path $VaultRoot "raw")
Ensure-Dir (Join-Path $VaultRoot "out")

# Target move map (conservative, “big media” only)
$targets = @(
  @{ src = Join-Path $RepoRoot "raw";                dst = Join-Path $VaultRoot "raw" },
  @{ src = Join-Path $RepoRoot "out\atomic_clips";   dst = Join-Path $VaultRoot "out\atomic_clips" },
  @{ src = Join-Path $RepoRoot "out\portrait_reels"; dst = Join-Path $VaultRoot "out\portrait_reels" },
  @{ src = Join-Path $RepoRoot "out\autoframe_work"; dst = Join-Path $VaultRoot "out\autoframe_work" },
  @{ src = Join-Path $RepoRoot "out\stabilized";     dst = Join-Path $VaultRoot "out\stabilized" }
)

# Report sizes before
Log "--- Size audit (GB) ---"
foreach ($t in $targets){
  $gb = Folder-SizeGB $t.src
  if ($gb -gt 0){
    Log ("{0,6} GB  {1}" -f $gb, $t.src)
  } else {
    Log ("{0,6} GB  {1} (missing/empty)" -f $gb, $t.src)
  }
}

# Perform planned moves (or just plan)
foreach ($t in $targets){
  Move-And-Junction $t.src $t.dst
}

# Ensure local always-on folders exist
Ensure-Dir (Join-Path $RepoRoot "out\telemetry")
Ensure-Dir (Join-Path $RepoRoot "out\logs")

# Gitignore update (do not force; append only if missing)
$gitignore = Join-Path $RepoRoot ".gitignore"
$needleLines = @(
  "",
  "# --- storage optimization: keep big media out of git ---",
  "raw/",
  "out/atomic_clips/",
  "out/portrait_reels/",
  "out/autoframe_work/",
  "out/stabilized/"
)

if (-not (Test-Path -LiteralPath $gitignore)){
  if ($Apply){
    Log "[do] creating .gitignore"
    Set-Content -LiteralPath $gitignore -Value ($needleLines -join "`r`n") -Encoding UTF8
  } else {
    Log "[plan] would create .gitignore with media ignores"
  }
} else {
  $existing = Get-Content -LiteralPath $gitignore -ErrorAction SilentlyContinue
  $missing = @()
  foreach ($l in $needleLines){
    if ($l -and ($existing -notcontains $l)) { $missing += $l }
  }
  if ($missing.Count -gt 0){
    if ($Apply){
      Log "[do] appending to .gitignore: $($missing -join ', ')"
      Add-Content -LiteralPath $gitignore -Value ("`r`n" + ($missing -join "`r`n")) -Encoding UTF8
    } else {
      Log "[plan] would append missing .gitignore lines: $($missing -join ', ')"
    }
  } else {
    Log "[ok] .gitignore already contains required media ignores"
  }
}

Log "=== Optimize-OneDriveVault done ==="
