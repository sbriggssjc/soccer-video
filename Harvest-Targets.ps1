<# 
  Harvest-Targets.ps1
  Build an exhaustive, deduped list of atomic clips from:
    - out\autoframe_work\cinematic (folders)
    - out\portrait_reels\branded   (files)

  Writes:
    - AtomicClips.All.csv  (everything we could infer)
    - AtomicClips.Todo.csv (items missing any stage)
  Optional: -Scaffold to create any missing cinematic folders for “branded-only” items.

  PowerShell 5-compatible (no null-coalescing, ternary, etc.)
#>

[CmdletBinding()]
param(
  [string]$RepoRoot      = ".",
  [string]$CinematicRoot = ".\out\autoframe_work\cinematic",
  [string]$BrandedRoot   = ".\out\portrait_reels\branded",
  [string]$OutAll        = ".\AtomicClips.All.csv",
  [string]$OutTodo       = ".\AtomicClips.Todo.csv",
  [switch]$Scaffold
)

# -------- Helpers --------
function Nz([string]$s, [string]$fallback = "") {
  if ([string]::IsNullOrWhiteSpace($s)) { return $fallback } else { return $s }
}
function JoinKey([string]$idx,[string]$date,[string]$homeTeam,[string]$awayTeam,[string]$label,[string]$tStart,[string]$tEnd) {
  # a stable composite key used across sources
  "$($idx)|$($date)|$($homeTeam)|$($awayTeam)|$($label)|$($tStart)|$($tEnd)"
}
function CleanBase([string]$s) {
  if (-not $s) { return $null }
  $x = $s -replace '\.__DEBUG(?:_FINAL)?_portrait_FINAL$',''
  $x = $x -replace '_portrait_FINAL$',''
  return $x
}
function Ensure-Dir([string]$p) {
  if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}

. "$PSScriptRoot\FolderRegex.ps1"
$FolderRegexPattern = $FolderRegex
$FolderRegexRx = [regex]$FolderRegexPattern

function Parse-CinematicFolder([string]$name) {
  $match = $FolderRegexRx.Match($name)
  if (-not $match.Success) { return $null }

  $date = if ($match.Groups['date'].Success) { $match.Groups['date'].Value } else { $null }
  if ([string]::IsNullOrWhiteSpace($date)) { $date = $null }

  $home = if ($match.Groups['home'].Success) { $match.Groups['home'].Value } else { $null }
  if ([string]::IsNullOrWhiteSpace($home)) { $home = $null }

  $away = if ($match.Groups['away'].Success) { $match.Groups['away'].Value } else { $null }
  if ([string]::IsNullOrWhiteSpace($away)) { $away = $null }

  $t1 = $match.Groups['t1'].Value
  $t2 = $match.Groups['t2'].Value

  $tStartRaw = "t$($t1)"
  $tEndRaw   = "t$($t2)"
  $timeTokenMatch = [regex]::Match($name, '__t(?<first>\d+(?:\.\d+)?)(?<sep>[-_])(?<second>t?\d+(?:\.\d+)?)')
  if ($timeTokenMatch.Success) {
    $tStartRaw = "t$($timeTokenMatch.Groups['first'].Value)"
    $secondRaw = $timeTokenMatch.Groups['second'].Value
    if ([string]::IsNullOrWhiteSpace($secondRaw)) {
      $tEndRaw = "t$($t2)"
    }
    elseif ($secondRaw.StartsWith('t')) {
      $tEndRaw = $secondRaw
    }
    else {
      $tEndRaw = "t$secondRaw"
    }
  }

  [pscustomobject]@{
    idx       = $match.Groups['idx'].Value
    date      = $date
    home      = $home
    away      = $away
    label     = $match.Groups['label'].Value
    t1        = $t1
    t2        = $t2
    tstartRaw = $tStartRaw
    tendRaw   = $tEndRaw
  }
}

# Branded file (with optional date/teams; BRAND or POST)
$rxBrand = [regex]'^(?<idx>\d{3})__(?:(?<date>\d{4}-\d{2}-\d{2})__(?<home>[^_]+)_vs_(?<away>[^_]+)__)?(?<label>[^_]+)__(?<tstart>t\d+\.\d{2})-(?<tend>t\d+\.\d{2})_portrait_(?<kind>BRAND|POST)\.mp4$'

# -------- Harvest cinematic (folders) --------
$cinRows = @()
if (Test-Path $CinematicRoot) {
  Get-ChildItem $CinematicRoot -Directory | ForEach-Object {
    $name = $_.Name
    $parsed = Parse-CinematicFolder $name
    if (-not $parsed) { return }  # unparsed; ignore

    $idx = $parsed.idx
    $date = $parsed.date
    $homeTeamName = $parsed.home
    $awayTeamName = $parsed.away
    $label = $parsed.label
    $tStart = $parsed.tstartRaw
    $tEnd   = $parsed.tendRaw

    $folderPath = $_.FullName
    $proxyPath  = Join-Path $folderPath 'proxy.mp4'
    $followDir  = Join-Path $folderPath 'follow'
    $trfPath    = Join-Path $followDir 'transforms.trf'
    $stabPath   = Join-Path $followDir 'stabilized.mp4'

    # Compute paths first (PowerShell 5-safe; no ternary)
    $proxyFound = Test-Path $proxyPath
    $trfFound   = Test-Path $trfPath
    $stabFound  = Test-Path $stabPath

    $proxyFinal = if ($proxyFound) { $proxyPath } else { $null }
    $trfFinal   = if ($trfFound)   { $trfPath   } else { $null }
    $stabFinal  = if ($stabFound)  { $stabPath  } else { $null }

    $cinRows += [pscustomobject]@{
      key               = JoinKey $idx (Nz $date) (Nz $homeTeamName) (Nz $awayTeamName) (Nz $label) (Nz $tStart) (Nz $tEnd)
      src               = 'cinematic'
      idx               = $idx
      date              = $date
      home              = $homeTeamName
      away              = $awayTeamName
      label             = $label
      t_start           = $tStart
      t_end             = $tEnd
      clip_folder       = $name
      clip_folder_path  = $folderPath
      proxy_path        = $proxyFinal
      transforms_path   = $trfFinal
      stabilized_path   = $stabFinal
      branded_files     = @()
    }
  }
}

# -------- Harvest branded (files) --------
$brandRows = @()
if (Test-Path $BrandedRoot) {
  Get-ChildItem $BrandedRoot -File -Filter "*.mp4" -Recurse | ForEach-Object {
    $name = $_.Name
    $m = $rxBrand.Match($name)
    if (-not $m.Success) { return }

    $idx  = $m.Groups['idx'].Value
    $date = $m.Groups['date'].Value
    $homeTeamName = $m.Groups['home'].Value
    $awayTeamName = $m.Groups['away'].Value
    $label = $m.Groups['label'].Value
    $tStart = $m.Groups['tstart'].Value
    $tEnd   = $m.Groups['tend'].Value
    $kind   = $m.Groups['kind'].Value  # BRAND or POST

    $brandRows += [pscustomobject]@{
      key               = JoinKey $idx (Nz $date) (Nz $homeTeamName) (Nz $awayTeamName) (Nz $label) (Nz $tStart) (Nz $tEnd)
      src               = 'branded'
      idx               = $idx
      date              = $date
      home              = $homeTeamName
      away              = $awayTeamName
      label             = $label
      t_start           = $tStart
      t_end             = $tEnd
      clip_folder       = $null
      clip_folder_path  = $null
      proxy_path        = $null
      transforms_path   = $null
      stabilized_path   = $null
      branded_files     = @([pscustomobject]@{ kind = $kind; path = $_.FullName })
    }
  }
}

# -------- Merge & reduce to unique keys --------
$byKey = @{}
foreach ($r in ($cinRows + $brandRows)) {
  if (-not $byKey.ContainsKey($r.key)) {
    # seed
    $byKey[$r.key] = [pscustomobject]@{
      key              = $r.key
      idx              = $r.idx
      date             = Nz $r.date
      home             = Nz $r.home
      away             = Nz $r.away
      label            = Nz $r.label
      t_start          = Nz $r.t_start
      t_end            = Nz $r.t_end
      clip_folder      = $r.clip_folder
      clip_folder_path = $r.clip_folder_path
      proxy_path       = $r.proxy_path
      transforms_path  = $r.transforms_path
      stabilized_path  = $r.stabilized_path
      branded_files    = @()
    }
  }

  $acc = $byKey[$r.key]

  # Prefer non-empty metadata
  if (-not $acc.clip_folder -and $r.clip_folder)      { $acc.clip_folder = $r.clip_folder }
  if (-not $acc.clip_folder_path -and $r.clip_folder_path) { $acc.clip_folder_path = $r.clip_folder_path }
  if (-not $acc.proxy_path -and $r.proxy_path)        { $acc.proxy_path = $r.proxy_path }
  if (-not $acc.transforms_path -and $r.transforms_path) { $acc.transforms_path = $r.transforms_path }
  if (-not $acc.stabilized_path -and $r.stabilized_path) { $acc.stabilized_path = $r.stabilized_path }

  # Aggregate branded files
  if ($r.src -eq 'branded' -and $r.branded_files) {
    $acc.branded_files += $r.branded_files
  }
}

# -------- Build final table with status --------
$all = @()
foreach ($kv in $byKey.GetEnumerator() | Sort-Object { $_.Value.idx, $_.Value.t_start }) {
  $v = $kv.Value
  $hasProxy      = [bool]( $v.proxy_path -and (Test-Path $v.proxy_path) )
  $hasTrans      = [bool]( $v.transforms_path -and (Test-Path $v.transforms_path) )
  $hasStab       = [bool]( $v.stabilized_path -and (Test-Path $v.stabilized_path) )
  $brandPaths    = @()
  $hasBrand      = $false
  $hasPost       = $false

  foreach ($bf in $v.branded_files) {
    if ($bf -and $bf.path) {
      $brandPaths += $bf.path
      if ($bf.kind -eq 'BRAND') { $hasBrand = $true }
      if ($bf.kind -eq 'POST')  { $hasPost  = $true }
    }
  }

  $all += [pscustomobject]@{
    key               = $v.key
    idx               = $v.idx
    date              = $v.date
    home              = $v.home
    away              = $v.away
    label             = $v.label
    t_start           = $v.t_start
    t_end             = $v.t_end
    clip_folder       = $v.clip_folder
    clip_folder_path  = $v.clip_folder_path
    proxy_exists      = $hasProxy
    transforms_exists = $hasTrans
    stabilized_exists = $hasStab
    branded_exists    = ($hasBrand -or $hasPost)
    branded_brand     = $hasBrand
    branded_post      = $hasPost
    branded_paths     = ($brandPaths -join ';')
    proxy_path        = $v.proxy_path
    transforms_path   = $v.transforms_path
    stabilized_path   = $v.stabilized_path
  }
}

# -------- Write outputs --------
$all | Sort-Object idx, t_start | Export-Csv -NoTypeInformation -Encoding UTF8 $OutAll

$todo = $all | Where-Object {
  -not $_.proxy_exists -or -not $_.transforms_exists -or -not $_.stabilized_exists -or -not $_.branded_exists
}
$todo | Sort-Object idx, t_start | Export-Csv -NoTypeInformation -Encoding UTF8 $OutTodo

# -------- Optional: scaffold missing cinematic folders from branded-only evidence --------
if ($Scaffold) {
  Write-Host "`nScaffolding missing cinematic folders..." -ForegroundColor Yellow
  Ensure-Dir $CinematicRoot

  $scaffolded = 0
  foreach ($r in $all | Where-Object { -not $_.clip_folder_path -and $_.branded_exists }) {
    # Build a best-effort folder name (prefer date/teams if present)
    $labelPart = Nz $r.label 'CLIP'
    $datePart  = Nz $r.date  ''
    $teamsPart = ''
    if ($r.home -and $r.away) { $teamsPart = "$($r.home)_vs_$($r.away)"; }
    $timePart  = "$($r.t_start)-$($r.t_end)"

    $base =
      if ($datePart -and $teamsPart) { "{0}__{1}__{2}__{3}__{4}" -f $r.idx,$datePart,$teamsPart,$labelPart,$timePart }
      else                           { "{0}__{1}__{2}" -f $r.idx,$labelPart,$timePart }

    $folderName = $base + "_portrait_FINAL"
    $target     = Join-Path $CinematicRoot $folderName

    # ensure uniqueness
    $suffix = 65 # 'A'
    while (Test-Path $target) {
      $folderName = $base + "_portrait_FINAL_" + [char]$suffix
      $target     = Join-Path $CinematicRoot $folderName
      $suffix++
    }

    Ensure-Dir $target
    Ensure-Dir (Join-Path $target 'follow')
    New-Item -ItemType File -Force -Path (Join-Path $target 'follow\.gitkeep') | Out-Null
    Write-Host "Created: $target"
    $scaffolded++
  }
  Write-Host "Scaffolded: $scaffolded" -ForegroundColor Green
}

# -------- Summary --------
[int]$nAll  = ($all | Measure-Object).Count
[int]$nTodo = ($todo | Measure-Object).Count
[int]$nBrandOnly = ($all | Where-Object { $_.branded_exists -and -not $_.clip_folder_path }).Count

Write-Host "`n=== Harvest Summary ===" -ForegroundColor Cyan
Write-Host ("Total unique targets:   {0}" -f $nAll)
Write-Host ("Missing any stage:      {0}" -f $nTodo)
Write-Host ("Branded-only (no cine): {0}" -f $nBrandOnly)
Write-Host ("All  -> {0}" -f (Resolve-Path $OutAll))
Write-Host ("TODO -> {0}" -f (Resolve-Path $OutTodo))
