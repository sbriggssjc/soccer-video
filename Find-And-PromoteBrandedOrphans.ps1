# Find-And-PromoteBrandedOrphans.ps1
# Scan branded reels on disk and promote any clips that aren't represented in cinematic
# by creating (or filling) canonical cinematic folders and hard-linking follow\stabilized.mp4

param(
  [string]$CineRoot    = ".\out\autoframe_work\cinematic",
  [string]$BrandedRoot = ".\out\portrait_reels\branded"
)

$ErrorActionPreference = 'Stop'

function Resolve-Dir([string]$p){
  if(-not (Test-Path $p)){ throw "Missing folder: $p" }
  (Resolve-Path $p).Path
}

$CineRoot    = Resolve-Dir $CineRoot
$BrandedRoot = Resolve-Dir $BrandedRoot

function Nz($v,$fallback=''){ if($null -ne $v -and "$v".Trim() -ne ''){ $v } else { $fallback } }
function SafeInt($v,$d=0){ try { [int]$v } catch { $d } }

# Parse long or short branded names to parts
# Supports:
#  001__2025-09-13__TSC_vs_NEOFC__GOAL__t180.80-t191.20_portrait_POST.mp4
#  002__GOAL__t180.80-t191.20_portrait_BRAND.mp4
$rxBranded = [regex]'^(?:(?<idx>\d{3})__)?(?:(?<date>\d{4}-\d{2}-\d{2})__)?(?:(?<home>[A-Za-z0-9_]+)_vs_(?<away>[A-Za-z0-9_]+)__)?(?<label>[A-Z0-9_]+)__t(?<t1>\d+(?:\.\d+)?)\-t(?<t2>\d+(?:\.\d+)?)_portrait_(?<kind>POST|BRAND|FINAL)\.mp4$'

# Parse cinematic folder names
#  001__2025-09-13__TSC_vs_NEOFC__GOAL__t180.80-t191.20_portrait_FINAL
$rxCine = [regex]'^(?<idx>\d{3})__(?<date>\d{4}-\d{2}-\d{2})__(?<home>[A-Za-z0-9_]+)_vs_(?<away>[A-Za-z0-9_]+)__(?<label>[A-Z0-9_]+)__t(?<t1>\d+(?:\.\d+)?)\-t(?<t2>\d+(?:\.\d+)?)_portrait_FINAL$'

function Parse-CineFolder([string]$name){
  $m = $rxCine.Match($name)
  if(-not $m.Success){ return $null }
  [pscustomobject]@{
    idx   = SafeInt $m.Groups['idx'].Value
    date  = $m.Groups['date'].Value
    home  = $m.Groups['home'].Value
    away  = $m.Groups['away'].Value
    label = $m.Groups['label'].Value
    t1    = $m.Groups['t1'].Value
    t2    = $m.Groups['t2'].Value
    key   = '{0}|{1}|{2}' -f $m.Groups['label'].Value,$m.Groups['t1'].Value,$m.Groups['t2'].Value
  }
}

function Parse-BrandedFile([IO.FileInfo]$fi){
  $m = $rxBranded.Match($fi.Name)
  if(-not $m.Success){ return $null }

  $idx      = if($m.Groups['idx'].Success){ SafeInt $m.Groups['idx'].Value } else { $null }
  $date     = Nz $m.Groups['date'].Value
  $homeTeam = Nz $m.Groups['home'].Value
  $awayTeam = Nz $m.Groups['away'].Value
  $label    = Nz $m.Groups['label'].Value
  $t1       = $m.Groups['t1'].Value
  $t2       = $m.Groups['t2'].Value
  $kind     = $m.Groups['kind'].Value

  [pscustomobject]@{
    idx=$idx; date=$date; home=$homeTeam; away=$awayTeam; label=$label; t1=$t1; t2=$t2; kind=$kind;
    key  = '{0}|{1}|{2}' -f $label,$t1,$t2
    path = $fi.FullName
    name = $fi.Name
  }
}

# Build a map of existing cinematic clips by key
$cineMap = @{}
$maxIdx  = 0
Get-ChildItem $CineRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  $info = Parse-CineFolder $_.Name
  if($null -eq $info){ return }
  if($info.idx -gt $maxIdx){ $maxIdx = $info.idx }
  $cineMap[$info.key] = [pscustomobject]@{
    folder     = $_.FullName
    name       = $_.Name
    idx        = $info.idx
    date       = $info.date
    home       = $info.home
    away       = $info.away
    label      = $info.label
    t1         = $info.t1
    t2         = $info.t2
    stabPath   = (Join-Path $_.FullName 'follow\stabilized.mp4')
  }
}

# Group branded files by key and prefer POST over BRAND over FINAL
$priority = @{ 'POST'=1; 'BRAND'=2; 'FINAL'=3 }
$brandedGroups = Get-ChildItem $BrandedRoot -File -Recurse -ErrorAction SilentlyContinue |
  ForEach-Object { Parse-BrandedFile $_ } |
  Where-Object { $_ -ne $null } |
  Group-Object key | ForEach-Object {
    $best = $_.Group | Sort-Object { $priority[$_.kind] } | Select-Object -First 1
    [pscustomobject]@{
      key   = $_.Name
      best  = $best
      all   = $_.Group
    }
  }

$created = 0
$linked  = 0
$skipped = 0

foreach($g in $brandedGroups){
  $best = $g.best
  $existing = $cineMap[$g.key]

  if($existing){
    $stab = $existing.stabPath
    if(Test-Path $stab){
      # already exists; check if already linked to same content
      # fast path: sizes equal and hardlink list contains both
      $same = $false
      try {
        $a = Get-Item $stab
        $b = Get-Item $best.path
        if($a.Length -eq $b.Length){
          $links = cmd /c fsutil hardlink list "$($best.path)" 2>$null
          if("$links" -match [regex]::Escape((Resolve-Path $stab).Path)){ $same = $true }
        }
      } catch { }
      if($same){ $skipped++; continue }
      # replace target with hardlink to best
      Remove-Item $stab -Force
      fsutil hardlink create "$stab" "$($best.path)" | Out-Null
      $linked++
    } else {
      New-Item -ItemType Directory -Force -Path (Split-Path $stab) | Out-Null
      fsutil hardlink create "$stab" "$($best.path)" | Out-Null
      $linked++
    }
    continue
  }

  # No existing cinematic: create one
  $maxIdx++
  $idxStr = '{0:000}' -f $maxIdx

  $date     = Nz $best.date '1970-01-01'
  $homeTeam = Nz $best.home 'HOME'
  $awayTeam = Nz $best.away 'AWAY'
  $label    = Nz $best.label 'CLIP'
  $t1       = $best.t1
  $t2       = $best.t2

  $folderName = '{0}__{1}__{2}_vs_{3}__{4}__t{5}-{6}_portrait_FINAL' -f $idxStr,$date,$homeTeam,$awayTeam,$label,$t1,$t2
  $cineFolder = Join-Path $CineRoot $folderName
  $follow     = Join-Path $cineFolder 'follow'
  $stab       = Join-Path $follow 'stabilized.mp4'

  New-Item -ItemType Directory -Force -Path $follow | Out-Null
  fsutil hardlink create "$stab" "$($best.path)" | Out-Null
  $cineMap[$g.key] = $true
  $created++
}

Write-Host ""
Write-Host "=== Promote Summary ===" -ForegroundColor Cyan
Write-Host ("Branded keys found:    {0}" -f $brandedGroups.Count)
Write-Host ("Created folders:       {0}" -f $created)
Write-Host ("Linked/updated stab:   {0}" -f $linked)
Write-Host ("Already satisfied:     {0}" -f $skipped)
Write-Host ""
Write-Host "Re-scan next:  powershell -NoProfile -ExecutionPolicy Bypass -File .\Scan-AtomicClips.ps1" -ForegroundColor Cyan
