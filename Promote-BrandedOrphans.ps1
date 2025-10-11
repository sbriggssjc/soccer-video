$cineRoot = Join-Path (Resolve-Path .).Path 'out\autoframe_work\cinematic'
$brandedOrphansPath = '.\AtomicClips.BranedOrphans.csv'

if (-not (Test-Path $brandedOrphansPath)) {
  Write-Host "Missing $brandedOrphansPath. Run Harvest-Targets/Status first." -ForegroundColor Red
  exit 1
}

# Get current max index (3-digit leading number in existing cine folders)
$existing = Get-ChildItem $cineRoot -Directory -ErrorAction SilentlyContinue
$maxIdx = 0
foreach ($d in $existing) {
  if ($d.Name -match '^(?<n>\d{3})__') {
    $n = [int]$Matches['n']; if ($n -gt $maxIdx) { $maxIdx = $n }
  }
}

function Nz($v,$fallback) { if ($null -ne $v -and $v -ne '') { $v } else { $fallback } }
function Q([string]$p) { if (-not $p) { return $null }; if ($p.Contains(' ')) { return '"' + $p + '"' } $p }

# Prefer POST, fallback to first file
function Pick-BrandedFile([string]$semiList) {
  if (-not $semiList) { return $null }
  $items = $semiList -split ';' | Where-Object { $_ -and (Test-Path $_) }
  if (-not $items) { return $null }
  $post = $items | Where-Object { $_ -like '*_portrait_POST.mp4' } | Select-Object -First 1
  if ($post) { return $post }
  return ($items | Select-Object -First 1)
}

$orphans = Import-Csv $brandedOrphansPath
$made = 0

foreach ($r in $orphans) {
  # Skip if we somehow already have a cinematic folder for this row
  if ($r.clip_folder) { continue }

  $srcFile = Pick-BrandedFile $r.branded_files
  if (-not $srcFile) {
    Write-Host "Skip: no branded_files path exists on disk for key $($r.key)" -ForegroundColor DarkYellow
    continue
  }

  $maxIdx++
  $idx = '{0:000}' -f $maxIdx

  $date = Nz $r.date '1970-01-01'
  $home = Nz $r.home 'HOME'
  $away = Nz $r.away 'AWAY'
  $label= Nz $r.label 'CLIP'

  # normalize times (strip non-digits, keep dot)
  $ts  = '{0}-{1}' -f ( (Nz $r.t_start '0') -replace '[^\d\.]','' ),
                     ( (Nz $r.t_end   '0') -replace '[^\d\.]','' )

  $folderName = '{0}__{1}__{2}_vs_{3}__{4}__t{5}_portrait_FINAL' -f $idx,$date,$home,$away,$label,$ts
  $cineFolder = Join-Path $cineRoot $folderName
  $follow     = Join-Path $cineFolder 'follow'
  $stab       = Join-Path $follow 'stabilized.mp4'

  New-Item -ItemType Directory -Force -Path $follow | Out-Null
  # Hard-link stabilized to branded source
  fsutil hardlink create (Q $stab) (Q $srcFile) | Out-Null
  $made++
  Write-Host "Scaffolded: $folderName" -ForegroundColor Green
}

Write-Host "`nCreated $made cinematic folder(s) from branded orphans." -ForegroundColor Cyan
Write-Host "Now re-scan: powershell -NoProfile -ExecutionPolicy Bypass -File .\Scan-AtomicClips.ps1" -ForegroundColor Cyan
