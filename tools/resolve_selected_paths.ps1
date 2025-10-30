Param(
  [string]$SrcCsv = "out\events_selected_resolved.csv",
  [string]$ClipsRoot = "out\atomic_clips",
  [string]$OutCsv = "out\events_selected_resolved.paths.csv",
  [double]$TolSec = 2.5    # time tolerance to declare a match
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Normalize-Time([double]$v){
  if($v -gt 1000){ $v = $v / 60.0 }     # convert framecount@60fps to seconds
  [math]::Round($v, 2)
}

function Parse-Stem([string]$stem){
  if(-not $stem){ return $null }
  # strip any debug suffix
  $s = $stem -replace '\.__DEBUG_FINAL$', ''
  # split at __t
  $parts = $s -split '__t'
  if($parts.Count -ne 2){ return $null }
  $prefix = $parts[0]
  if($parts[1] -notmatch '^([0-9.]+)-t([0-9.]+)$'){ return $null }
  $t1 = Normalize-Time([double]$Matches[1])
  $t2 = Normalize-Time([double]$Matches[2])
  [pscustomobject]@{ Prefix=$prefix; T1=$t1; T2=$t2 }
}

function Stem-From-Path([string]$path){
  [System.IO.Path]::GetFileNameWithoutExtension($path)
}

function Build-Inventory($root){
  Write-Host "Scanning $root for .mp4 ..." -ForegroundColor Cyan
  $files = Get-ChildItem -Recurse -LiteralPath $root -Filter *.mp4 -ErrorAction SilentlyContinue
  if(-not $files){ throw "No .mp4 files found under $root" }

  $map = @{}   # prefix -> list of {T1,T2,Stem,Path}
  foreach($f in $files){
    $stem = Stem-From-Path $f.FullName
    $p = Parse-Stem $stem
    if(-not $p){ continue }
    $entry = [pscustomobject]@{ Prefix=$p.Prefix; T1=$p.T1; T2=$p.T2; Stem=$stem; Path=$f.FullName }
    if(-not $map.ContainsKey($p.Prefix)){ $map[$p.Prefix] = New-Object System.Collections.Generic.List[object] }
    $map[$p.Prefix].Add($entry)
  }
  Write-Host ("Inventory: {0} files, {1} prefixes" -f $files.Count, $map.Keys.Count)
  return $map
}

function Best-Match($inv, [string]$csvStem, [double]$tolSec){
  $parsed = Parse-Stem $csvStem
  if(-not $parsed){ return $null }
  if(-not $inv.ContainsKey($parsed.Prefix)){ return $null }

  $cands = $inv[$parsed.Prefix]

  # distance = max(|Δstart|, |Δend|)
  $scored = foreach($c in $cands){
    $d1 = [math]::Abs($c.T1 - $parsed.T1)
    $d2 = [math]::Abs($c.T2 - $parsed.T2)
    $d  = [math]::Max($d1,$d2)
    [pscustomobject]@{ D=$d; Cand=$c }
  }

  $best = $scored | Sort-Object D | Select-Object -First 1
  if(-not $best){ return $null }
  if($best.D -le $tolSec){ return $best.Cand }
  return $null
}

# --- MAIN ---
$pwd = (Get-Location).Path
Write-Host ("PWD: {0}" -f $pwd)
if(-not (Test-Path -LiteralPath $SrcCsv)){ throw "Missing CSV: $SrcCsv" }
if(-not (Test-Path -LiteralPath $ClipsRoot)){ throw "Missing clips folder: $ClipsRoot" }

$rows = Import-Csv -LiteralPath $SrcCsv
if(-not $rows -or $rows.Count -eq 0){ throw "CSV has no rows: $SrcCsv" }

# Discover which column holds paths/stems
$cols = $rows | Get-Member -MemberType NoteProperty | Select-Object -Expand Name
$mp4Col = @('mp4','clip','video','path') | Where-Object { $cols -contains $_ } | Select-Object -First 1
if(-not $mp4Col){ throw "No mp4-like column found in CSV. Columns: $($cols -join ', ')" }
Write-Host ("Using column: {0}" -f $mp4Col)

$inv = Build-Inventory $ClipsRoot

$matched = New-Object System.Collections.Generic.List[object]
$unresolved = New-Object System.Collections.Generic.List[object]

foreach($r in $rows){
  $raw = [string]$r.$mp4Col
  if(-not $raw){ continue }
  # normalize incoming value -> stem
  $leaf = Split-Path $raw -Leaf
  $stem = [System.IO.Path]::GetFileNameWithoutExtension($leaf)
  $hit = Best-Match -inv $inv -csvStem $stem -tolSec $TolSec
  if($hit){
    $matched.Add([pscustomobject]@{ mp4 = $hit.Path })
  } else {
    $unresolved.Add($stem)
  }
}

if($matched.Count -eq 0){
  Write-Warning "No rows could be resolved. Try increasing -TolSec."
  if($unresolved.Count){ "First unresolved:"; $unresolved | Select-Object -First 10 | ForEach-Object { "  - $_" } }
  throw "Stopping."
}

$null = New-Item -ItemType Directory -Path (Split-Path -Parent $OutCsv) -ErrorAction SilentlyContinue
$matched | Sort-Object mp4 -Unique | Export-Csv -LiteralPath $OutCsv -NoTypeInformation -Encoding UTF8

Write-Host ("[OK] Resolved {0}/{1} rows -> {2}" -f $matched.Count, $rows.Count, $OutCsv) -ForegroundColor Green

if($unresolved.Count){
  Write-Host ("Unresolved stems (showing up to 12):") -ForegroundColor Yellow
  $unresolved | Select-Object -First 12 | ForEach-Object { "  - $_" }
  Write-Host "Tip: re-run with a larger -TolSec (e.g., -TolSec 4.0) if times were trimmed/rounded."
}
