param(
  [string]$IndexCsv   = ".\atomic_clips_index.rebuilt.csv",
  [string]$AtomicRoot = ".\out\atomic_clips",
  [switch]$DoCut,
  [switch]$ForceRecut,
  [ValidateSet("stabilized","trf","proxy")]
  [string]$PreferredSource = "stabilized"
)

function Slug([string]$s){
  if ([string]::IsNullOrWhiteSpace($s)) { return "" }
  $t = $s -replace '\s+','_'
  $t = $t -replace '[^A-Za-z0-9_\-]',''
  return $t
}
function Parse-T([string]$t) {
  if ([string]::IsNullOrWhiteSpace($t)) { return $null }
  $u = ($t -replace '^[tT]','').Trim()
  [double]::TryParse($u, [ref]([double]$null)) | Out-Null
  return [double]$u
}
function File-Exists([string]$p){ if ([string]::IsNullOrWhiteSpace($p)) { return $false } Test-Path -LiteralPath $p -PathType Leaf }
function Choose-Master($r, [string]$pref){
  $candidates = @()
  if ($pref -eq "stabilized") { $candidates = @($r.stabilized_path, $r.trf_path, $r.proxy_path) }
  elseif ($pref -eq "trf")    { $candidates = @($r.trf_path, $r.stabilized_path, $r.proxy_path) }
  else                         { $candidates = @($r.proxy_path, $r.stabilized_path, $r.trf_path) }
  foreach($p in $candidates){ if (File-Exists $p) { return $p } }
  return $null
}

if (-not (Test-Path -LiteralPath $IndexCsv)) { throw "Index CSV not found: $IndexCsv" }
$rows = Import-Csv -LiteralPath $IndexCsv

# pre-scan existing atomics by index prefix "NNN__"
$haveByIdx = New-Object 'System.Collections.Generic.HashSet[string]'
if (Test-Path -LiteralPath $AtomicRoot) {
  Get-ChildItem -LiteralPath $AtomicRoot -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^\d{3}__' } |
    ForEach-Object {
      $m = [regex]::Match($_.Name, '^(?<idx>\d{3})__')
      if ($m.Success) { [void]$haveByIdx.Add($m.Groups['idx'].Value) }
    }
}

$plan = New-Object System.Collections.Generic.List[object]
$have = 0; $miss = 0

foreach ($r in $rows) {
  $idxStr = ([string]$r.index).Trim()
  if (-not $idxStr) { continue }
  $idx3 = '{0:d3}' -f ([int]$idxStr)
  $date = [string]$r.date
  $homeTeamName = Slug $r.homeTeam
  $awayTeamName = Slug $r.awayTeam
  $label = if ($r.label) { Slug $r.label } else { "CLIP" }
  $tS = Parse-T $r.tStart
  $tE = Parse-T $r.tEnd
  $tS2 = if ($tS -ne $null) { $tS } else { 0 }
  $tE2 = if ($tE -ne $null) { $tE } else { 0 }

  $folder = Join-Path $AtomicRoot ("{0}__{1}_vs_{2}" -f $date,$homeTeamName,$awayTeamName)
  $fname  = ("{0}__{1}__{2}_vs_{3}__{4}__t{5:F2}-t{6:F2}.mp4" -f $idx3,$date,$homeTeamName,$awayTeamName,$label,$tS2,$tE2) -replace '\.00\b',''
  $outPath = Join-Path $folder $fname

  $existsForIndex = $haveByIdx.Contains($idx3)
  if ($existsForIndex -and -not $ForceRecut) {
    $have++
    $plan.Add([pscustomobject]@{ index=$idxStr; date=$date; status='HAVE'; atomic=$null; master=$null }) | Out-Null
    continue
  }

  $master = Choose-Master $r $PreferredSource
  if ($null -eq $master) {
    $miss++
    $plan.Add([pscustomobject]@{ index=$idxStr; date=$date; status='MISSING: no master'; atomic=$outPath; master=$null }) | Out-Null
    continue
  }

  $status = if ($ForceRecut) { 'RECUT' } else { 'MISSING' }
  $miss++
  $plan.Add([pscustomobject]@{ index=$idxStr; date=$date; status=$status; atomic=$outPath; master=$master }) | Out-Null

  if ($DoCut) {
    if (-not (Test-Path -LiteralPath $folder)) { New-Item -ItemType Directory -Path $folder -Force | Out-Null }
    if (($tS -ne $null) -and ($tE -ne $null) -and ($tE -gt $tS)) {
      $dur = [math]::Max(0.02, $tE - $tS)
      $ff = "ffmpeg"
      $args = @(
        "-y","-hide_banner","-loglevel","error",
        "-ss",("{0:F3}" -f $tS), "-i",$master, "-t",("{0:F3}" -f $dur),
        "-c:v","libx264","-preset","veryfast","-crf","18",
        "-c:a","aac","-b:a","128k",
        $outPath
      )
      & $ff $args 2>$null
    }
  }
}

$stamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
$outReport = ".\atomic_clips_audit_{0}.csv" -f $stamp
$plan | Export-Csv -LiteralPath $outReport -NoTypeInformation -Encoding UTF8

"Rows: $($rows.Count)"
"HAVE atomic:       $have"
"Missing (plan):    $miss"
"Report -> $outReport"

$plan | Select-Object -First 24 index,date,status,atomic | Format-Table
