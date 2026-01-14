[CmdletBinding(SupportsShouldProcess)]
param(
  [string]$VaultRoot = "C:\Users\scott\OneDrive\SoccerVideoMedia",
  [int]$Days = 14,
  [int]$SkipModifiedHours = 24,
  [string[]]$Extensions = @("mp4","mov","mxf","mkv","avi","wav","mp3"),
  [string]$RepoRoot = "C:\Users\scott\soccer-video"
)

$ErrorActionPreference = "Continue"
$LogDir = Join-Path $RepoRoot "out\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir "storage_offload.log"

function Log($msg){
  $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $line = "$ts $msg"
  $line | Tee-Object -FilePath $LogPath -Append | Out-Host
}

function Try-Offload([string]$file){
  # Best-effort methods; behavior varies by OneDrive build.
  # attrib +U  : mark as unpinned/“online-only capable”
  # attrib -P  : remove pinned “always keep on this device”
  # Some systems accept these; some ignore them. We log result either way.
  $ok = $true
  try {
    cmd /c "attrib +U -P `"$file`"" | Out-Null
  } catch {
    $ok = $false
  }
  return $ok
}

Log "=== Offload-OneDriveMedia starting (Days=$Days SkipModifiedHours=$SkipModifiedHours) ==="
Log "VaultRoot: $VaultRoot"

if (-not (Test-Path -LiteralPath $VaultRoot)){
  Log "[error] VaultRoot not found: $VaultRoot"
  exit 1
}

$cutoffOld = (Get-Date).AddDays(-$Days)
$cutoffHot = (Get-Date).AddHours(-$SkipModifiedHours)

$extSet = New-Object "System.Collections.Generic.HashSet[string]" ([StringComparer]::OrdinalIgnoreCase)
$Extensions | ForEach-Object { $extSet.Add($_.TrimStart(".")) | Out-Null }

$files = Get-ChildItem -LiteralPath $VaultRoot -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object {
    $e = $_.Extension.TrimStart(".")
    $extSet.Contains($e)
  }

$scanned = 0
$eligible = 0
$offloaded = 0
$skippedHot = 0
$skippedNew = 0
$failed = 0

foreach ($f in $files){
  $scanned++

  if ($f.LastWriteTime -gt $cutoffHot){
    $skippedHot++
    continue
  }
  if ($f.LastWriteTime -gt $cutoffOld){
    $skippedNew++
    continue
  }

  $eligible++

  if ($PSCmdlet.ShouldProcess($f.FullName, "Mark online-only/unpinned")){
    $ok = Try-Offload $f.FullName
    if ($ok){
      $offloaded++
    } else {
      $failed++
      Log "[fail] attrib method failed: $($f.FullName)"
    }
  }
}

Log "Summary: scanned=$scanned eligible=$eligible offloaded=$offloaded skippedHot=$skippedHot skippedTooNew=$skippedNew failed=$failed"
Log "=== Offload-OneDriveMedia done ==="
