# Promote-BrandedOrphans.ps1
# Promote branded-only rows (clip_folder == "(BRANDED_ORPHAN)" or blank) into canonical cinematic folders
# and hard-link follow\stabilized.mp4 to the preferred branded POST file.

param(
  [string]$CineRoot    = ".\out\autoframe_work\cinematic",
  [string]$BrandedRoot = ".\out\portrait_reels\branded",
  [string]$OrphansCsv  = ".\AtomicClips.BranedOrphans.csv"   # file name uses repo’s current spelling
)

$CineRoot    = (Resolve-Path $CineRoot).Path
$BrandedRoot = (Resolve-Path $BrandedRoot).Path
if (-not (Test-Path $OrphansCsv)) { Write-Host "Missing $OrphansCsv" -ForegroundColor Red; exit 1 }

function Nz($v,$fallback){ if($null -ne $v -and "$v".Trim() -ne ''){ $v } else { $fallback } }
function SafeInt($v,$d=0){ try { [int]$v } catch { $d } }

# Find preferred branded file for a row
function Get-PreferredBrandedFile([object]$r){
  # 1) explicit columns
  foreach($col in 'branded_path','branded_file'){
    if($r.PSObject.Properties.Match($col) -and $r.$col -and (Test-Path $r.$col)){ return $r.$col }
  }
  if($r.PSObject.Properties.Match('branded_files') -and $r.branded_files){
    $list = $r.branded_files -split ';' | Where-Object { $_ -and (Test-Path $_) }
    if($list){ 
      $post = $list | Where-Object { $_ -like '*_portrait_POST.mp4' } | Select-Object -First 1
      if($post){ return $post }
      return ($list | Select-Object -First 1)
    }
  }
  # 2) search by key hint in BrandedRoot
  $key = $null
  foreach($k in 'key','short_key','base_key'){
    if($r.PSObject.Properties.Match($k) -and $r.$k){ $key = $r.$k; break }
  }
  if($key){
    $candidates = Get-ChildItem $BrandedRoot -Filter "*$key*portrait_*.mp4" -File -ErrorAction SilentlyContinue
    if($candidates){
      $post = $candidates | Where-Object { $_.Name -like '*_portrait_POST.mp4' } | Select-Object -First 1
      if($post){ return $post.FullName }
      return ($candidates | Select-Object -First 1).FullName
    }
  }
  # 3) last resort: try to reconstruct a loose pattern from label/times
  $label = Nz $r.label ''
  $ts    = '{0}-{1}' -f ( (Nz $r.t_start '0') -replace '[^\d\.]','' ),
                      ( (Nz $r.t_end   '0') -replace '[^\d\.]','' )
  if($label -or $ts){
    $pattern = "*${label}*t${ts}*_portrait_*.mp4" -replace '\*{2,}','*'
    $candidates = Get-ChildItem $BrandedRoot -Filter $pattern -File -ErrorAction SilentlyContinue
    if($candidates){
      $post = $candidates | Where-Object { $_.Name -like '*_portrait_POST.mp4' } | Select-Object -First 1
      if($post){ return $post.FullName }
      return ($candidates | Select-Object -First 1).FullName
    }
  }
  return $null
}

# current max 3-digit index in existing cinematic folders
$maxIdx = 0
Get-ChildItem $CineRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  if($_.Name -match '^(?<n>\d{3})__'){ $n = SafeInt $Matches['n'] 0; if($n -gt $maxIdx){ $maxIdx = $n } }
}

$rows   = Import-Csv $OrphansCsv
$target = $rows | Where-Object {
  $f = Nz $_.clip_folder ''
  ($f -eq '') -or ($f -eq '(BRANDED_ORPHAN)')
}

if(-not $target){ Write-Host "No branded-only rows found in $OrphansCsv." -ForegroundColor Yellow; exit 0 }

$made = 0
foreach($r in $target){
  $src = Get-PreferredBrandedFile $r
  if(-not $src){ 
    Write-Host "Skip (no branded file on disk): key=$($r.key)" -ForegroundColor DarkYellow
    continue
  }

  $maxIdx++
  $idx = '{0:000}' -f $maxIdx

  $date = Nz $r.date '1970-01-01'
  $home = Nz $r.home 'HOME'
  $away = Nz $r.away 'AWAY'
  $label= Nz $r.label 'CLIP'

  $tStart = (Nz $r.t_start '0') -replace '[^\d\.]',''
  $tEnd   = (Nz $r.t_end   '0') -replace '[^\d\.]',''
  $ts     = "$tStart-$tEnd"

  $folderName = '{0}__{1}__{2}_vs_{3}__{4}__t{5}_portrait_FINAL' -f $idx,$date,$home,$away,$label,$ts
  $cineFolder = Join-Path $CineRoot $folderName
  $follow     = Join-Path $cineFolder 'follow'
  $stab       = Join-Path $follow 'stabilized.mp4'

  if(Test-Path $stab){
    Write-Host "Exists, skip: $folderName" -ForegroundColor DarkGray
    continue
  }

  New-Item -ItemType Directory -Force -Path $follow | Out-Null
  fsutil hardlink create "$stab" "$src" | Out-Null

  $made++
  Write-Host "Scaffolded: $folderName  ⟵  $(Split-Path $src -Leaf)" -ForegroundColor Green
}

Write-Host "`nCreated $made cinematic folder(s) from branded orphans." -ForegroundColor Cyan
Write-Host "Re-scan: powershell -NoProfile -ExecutionPolicy Bypass -File .\Scan-AtomicClips.ps1" -ForegroundColor Cyan
