param(
  [Parameter(Mandatory=$true)] [string]$SrcCsv,
  [Parameter(Mandatory=$true)] [string]$ClipsRoot,
  [Parameter(Mandatory=$true)] [string]$OutCsv,
  [double]$TolSec = 4.0,
  [string]$CsvCol = "mp4"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Write-Host "PWD: $(Get-Location)"
Write-Host "Using column: $CsvCol"
Write-Host "Scanning $ClipsRoot for .mp4 ..."

function Parse-Num([string]$s){
  if(-not $s){ return $null }
  $s = $s -replace '[^0-9\.,-]',''
  $s = $s -replace ',',''
  if(($s.Split('.').Count -gt 2)){
    $parts = $s.Split('.')
    $s = $parts[0] + '.' + (($parts[1..($parts.Count-1)]) -join '')
  }
  return [double]::Parse($s, [Globalization.CultureInfo]::InvariantCulture)
}

function Normalize-Time([double]$v){
  if($v -gt 1000 -and ([math]::Abs(($v/60) - [math]::Round($v/60,2)) -lt 0.01)){
    $v = $v/60.0
  }
  [math]::Round($v, 2)
}

$StemRegex = '^(?<prefix>.+?)__t(?<t1>[-0-9\.,]+)-t(?<t2>[-0-9\.,]+)(?:\.__.*)?$'

function Parse-Stem([string]$nameNoExt){
  if(-not $nameNoExt){ return $null }
  if($nameNoExt -match $StemRegex){
    $prefix = $Matches['prefix']
    $n1 = Parse-Num $Matches['t1']; if($null -eq $n1){ return $null }
    $n2 = Parse-Num $Matches['t2']; if($null -eq $n2){ return $null }
    $t1 = Normalize-Time $n1
    $t2 = Normalize-Time $n2
    [pscustomobject]@{ Prefix=$prefix; T1=$t1; T2=$t2 }
  } else {
    $null
  }
}

function CanonPrefix([string]$p){
  if(-not $p){ return '' }
  ($p.ToLower() -replace '\s+',' ' -replace '_+','_').Trim('_ ')
}

function TailPrefix([string]$p, [int]$keep=3){
  if(-not $p){ return '' }
  $parts = $p -split '__'
  if($parts.Count -le $keep){ return CanonPrefix $p }
  $tail = ($parts[($parts.Count-$keep)..($parts.Count-1)]) -join '__'
  CanonPrefix $tail
}

$diskItems = @()
Get-ChildItem -LiteralPath $ClipsRoot -Recurse -Filter *.mp4 |
  ForEach-Object {
    $stem = $_.BaseName
    $ps = Parse-Stem $stem
    if($ps){
      $diskItems += [pscustomobject]@{
        Full   = $_.FullName
        Prefix = $ps.Prefix
        Canon  = CanonPrefix $ps.Prefix
        Tail2  = TailPrefix  $ps.Prefix 2
        Tail3  = TailPrefix  $ps.Prefix 3
        T1     = $ps.T1
        T2     = $ps.T2
      }
    }
  }

Write-Host ("Inventory: {0} files, {1} prefixes" -f ($diskItems.Count), (($diskItems | Select-Object -ExpandProperty Canon -Unique).Count))

$byCanon = $diskItems | Group-Object Canon -AsHashTable -AsString
$byTail2 = $diskItems | Group-Object Tail2 -AsHashTable -AsString
$byTail3 = $diskItems | Group-Object Tail3 -AsHashTable -AsString

$rows = Import-Csv -LiteralPath $SrcCsv
$need = @()
foreach($r in $rows){
  $raw = [string]$r.$CsvCol
  if(-not $raw){ continue }
  $name = [IO.Path]::GetFileNameWithoutExtension($raw -replace '"','')
  $ps = Parse-Stem $name
  if($ps){
    $need += [pscustomobject]@{
      Raw     = $raw
      Prefix  = $ps.Prefix
      Canon   = CanonPrefix $ps.Prefix
      Tail2   = TailPrefix  $ps.Prefix 2
      Tail3   = TailPrefix  $ps.Prefix 3
      T1      = $ps.T1
      T2      = $ps.T2
    }
  }
}

function Try-Resolve($wanted){
  $cands = @()
  if($byCanon.ContainsKey($wanted.Canon)){
    $cands += $byCanon[$wanted.Canon]
  }
  if(-not $cands -and $byTail3.ContainsKey($wanted.Tail3)){
    $cands += $byTail3[$wanted.Tail3]
  }
  if(-not $cands -and $byTail2.ContainsKey($wanted.Tail2)){
    $cands += $byTail2[$wanted.Tail2]
  }
  if(-not $cands){ return $null }

  $hits = $cands | Where-Object {
    ([math]::Abs($_.T1 - $wanted.T1) -le $TolSec) -and
    ([math]::Abs($_.T2 - $wanted.T2) -le $TolSec)
  }

  if($hits.Count -gt 0){
    $hits | Sort-Object @{E={[math]::Abs($_.T1-$wanted.T1)+[math]::Abs($_.T2-$wanted.T2)}} | Select-Object -First 1
  } else {
    $null
  }
}

$resolved = @()
$unres    = @()

foreach($w in $need){
  $hit = Try-Resolve $w
  if($hit){
    $resolved += [pscustomobject]@{ mp4 = $hit.Full }
  } else {
    $unres += $w
  }
}

if(-not $resolved -or $resolved.Count -eq 0){
  Write-Warning "No rows could be resolved. Try increasing -TolSec."
  if($unres.Count){
    "First unresolved:"
    $unres | Select-Object -First 10 | ForEach-Object {
      "  - $($_.Prefix)__t$($_.T1)-t$($_.T2)"
    }
  }
  throw "Stopping."
}

$dir = Split-Path -Parent $OutCsv
if($dir){ New-Item -ItemType Directory -Force -Path $dir | Out-Null }
$resolved | Export-Csv -LiteralPath $OutCsv -NoTypeInformation -Encoding UTF8
Write-Host ("[OK] Resolved {0} rows to real files -> {1}" -f $resolved.Count, $OutCsv)

$wantStems = ($need | ForEach-Object { "$($_.Prefix)__t$($_.T1)-t$($_.T2)" } | Select-Object -Unique).Count
$haveStems = ($diskItems | ForEach-Object { "$($_.Prefix)__t$($_.T1)-t$($_.T2)" } | Select-Object -Unique).Count
Write-Host ("Wanted (unique stems in CSV): {0}" -f $wantStems)
Write-Host ("Have   (unique stems on disk): {0}" -f $haveStems)
if($unres.Count){
  Write-Host "Stems in CSV not found on disk (first 10):"
  $unres | Select-Object -First 10 | ForEach-Object {
    "  - $($_.Prefix)__t$($_.T1)-t$($_.T2)"
  }
}
$unref = @()
$diskKey = @{}
$diskItems | ForEach-Object { $diskKey["$($_.Prefix)__t$($_.T1)-t$($_.T2)"]=1 }
$needKey = @{}
$need     | ForEach-Object { $needKey["$($_.Prefix)__t$($_.T1)-t$($_.T2)"]=1 }
$diskKey.Keys | ForEach-Object { if(-not $needKey.ContainsKey($_)){ $unref += $_ } }
if($unref.Count){
  Write-Host "Stems on disk not referenced by CSV (first 10):"
  $unref | Select-Object -First 10 | ForEach-Object { "  - $_" }
}
