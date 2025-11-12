param(
  [string]$Video = ".\out\full_game_stabilized.mp4",

  [ValidateSet("ignore","strict","loose")]
  [string]$GoalMode = "loose",

  [int]$MaxGoals = 99,
  [double]$MinGoalSeparation = 5.0,

  [double]$ActionPad = 0.8,
  [double]$GoalPadBefore = 2.0,
  [double]$GoalPadAfter  = 3.0,

  [double]$ActionMergeGap = 1.5,  # smoothness of action merging
  [double]$MinSpanSec     = 2.0,  # drop micro-clips
  [double]$MaxFractionOfRaw = 0.25
)

# ---------- helpers ----------
function TryParse-Double([object]$x, [ref]$out) {
  $s = "$x" -replace ',', ''
  [double]$tmp = 0
  $ok = [System.Double]::TryParse(
    $s,
    [System.Globalization.NumberStyles]::Float,
    [System.Globalization.CultureInfo]::InvariantCulture,
    [ref]$tmp
  )
  $out.Value = $tmp
  return $ok
}

function Get-VideoDurationSec([string]$path) {
  $durStr = & ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 $path
  [double]$d = 0; TryParse-Double $durStr ([ref]$d) | Out-Null
  return [math]::Round($d,3)
}

function Read-Spans([string]$csvPath, [double]$padBefore=0.0, [double]$padAfter=0.0, [string]$source="") {
  $sp=@()
  if (-not (Test-Path $csvPath)) { return $sp }
  $rows = Import-Csv $csvPath
  foreach ($r in $rows) {
    [double]$a=0; [double]$b=0
    if (TryParse-Double $r.start ([ref]$a) -and TryParse-Double $r.end ([ref]$b) -and $b -gt $a) {
      $t0 = [math]::Max(0,$a - $padBefore)
      $t1 = $b + $padAfter
      if ($t1 -gt $t0) { $sp += [pscustomobject]@{ t0=$t0; t1=$t1; src=$source } }
    }
  }
  return $sp
}

function Merge-Spans($spans, [double]$minGap=0.0) {
  $out=@()
  $ordered = $spans | Sort-Object t0, t1
  foreach ($s in $ordered) {
    if ($out.Count -eq 0) { $out += $s; continue }
    $last = $out[-1]
    if ($s.t0 -le $last.t1 + $minGap) {
      if ($s.t1 -gt $last.t1) { $last.t1 = $s.t1 }
    } else { $out += $s }
  }
  return $out
}

function Merge-GoalSpans($spans, [double]$minGap=0.0) {
  # like Merge-Spans but keeps max(score) when coalescing
  $out=@()
  $ordered = $spans | Sort-Object t0, t1
  foreach ($s in $ordered) {
    if ($out.Count -eq 0) { $out += $s; continue }
    $last = $out[-1]
    if ($s.t0 -le $last.t1 + $minGap) {
      if ($s.t1 -gt $last.t1) { $last.t1 = $s.t1 }
      if ($s.PSObject.Properties.Match('score').Count -gt 0) { 
        if ($last.PSObject.Properties.Match('score').Count -eq 0) { $last | Add-Member -NotePropertyName score -NotePropertyValue 0.0 }
        if ($s.score -gt $last.score) { $last.score = $s.score }
      }
    } else { $out += $s }
  }
  return $out
}

function Cap-ByFraction($spans, [double]$videoDur, [double]$maxFrac) {
  if (-not $spans -or $spans.Count -eq 0) { return @() }
  if ($maxFrac -le 0 -or $maxFrac -ge 1) { return $spans }
  $budget = $videoDur * $maxFrac
  if ($budget -le 0) { return @() }

  $picked=@(); $sum=0.0
  foreach ($s in ($spans | Sort-Object t0)) {
    $len = [math]::Max(0, $s.t1 - $s.t0)
    if ($len -le 0) { continue }

    if ($sum + $len -le $budget) {
      $picked += $s; $sum += $len
    } elseif ($picked.Count -eq 0) {
      # ensure at least one span
      $picked += $s
      break
    } else {
      break
    }
  }
  return $picked
}

function Get-GoalSpans([string]$outDir, [string]$mode, [double]$minSep, [int]$maxN) {
  if ($mode -eq "ignore") { return @() }

  $grCSV = Join-Path $outDir "goal_resets.csv"
  $fgCSV = Join-Path $outDir "forced_goals.csv"
  $cand=@()

  if (Test-Path $grCSV) {
    $rows = Import-Csv $grCSV
    foreach ($r in $rows) {
      [double]$a=0; [double]$b=0; [double]$sc=0
      if (TryParse-Double $r.start ([ref]$a) -and TryParse-Double $r.end ([ref]$b) -and $b -gt $a) {
        TryParse-Double $r.score ([ref]$sc) | Out-Null
        $cand += [pscustomobject]@{ t0=$a; t1=$b; score=$sc; src="goal" }
      }
    }
  }

  if ($mode -eq "loose" -and (Test-Path $fgCSV)) {
    $rows = Import-Csv $fgCSV
    foreach ($r in $rows) {
      [double]$a=0; [double]$b=0
      if (TryParse-Double $r.start ([ref]$a) -and TryParse-Double $r.end ([ref]$b) -and $b -gt $a) {
        $cand += [pscustomobject]@{ t0=$a; t1=$b; score=1.0; src="goal_forced" }
      }
    }
  }

  if ($cand.Count -eq 0) { return @() }

  $merged = Merge-GoalSpans $cand $minSep
  $ordered = $merged | Sort-Object -Property @{Expression="score";Descending=$true}, @{Expression="t0";Descending=$false}
  if ($maxN -gt 0) { $ordered = $ordered | Select-Object -First $maxN }
  return $ordered
}

# ---------- main ----------
$ErrorActionPreference = "Stop"
$OutDir = ".\out"
if (!(Test-Path $Video)) { throw "Missing video: $Video" }
$duration = Get-VideoDurationSec $Video

$playsCSV = Join-Path $OutDir "plays.csv"
$hi1CSV   = Join-Path $OutDir "highlights_shots.csv"
$hi2CSV   = Join-Path $OutDir "highlights_filtered.csv"

$goals = Get-GoalSpans $OutDir $GoalMode $MinGoalSeparation $MaxGoals
# pad goals (pre/post) and clamp within video duration
$goals = $goals | ForEach-Object {
  $t0 = [math]::Max(0, $_.t0 - $GoalPadBefore)
  $t1 = [math]::Min($duration, $_.t1 + $GoalPadAfter)

  # compute score without using a ternary (PS5.1-safe)
  $sc = 0.0
  if ($_.PSObject.Properties.Match('score').Count -gt 0) {
    [double]$tmp = 0
    if (TryParse-Double $_.score ([ref]$tmp)) { $sc = $tmp }
  }

  [pscustomobject]@{ t0 = $t0; t1 = $t1; src = $_.src; score = $sc }
}

$actions = @()
if (Test-Path $playsCSV) { $actions += Read-Spans $playsCSV $ActionPad $ActionPad "plays" }
if (Test-Path $hi2CSV)   { $actions += Read-Spans $hi2CSV $ActionPad $ActionPad "hi_filtered" }
if (Test-Path $hi1CSV)   { $actions += Read-Spans $hi1CSV $ActionPad $ActionPad "hi_shots" }
$actions = Merge-Spans $actions $ActionMergeGap

# drop actions that overlap any goal
$keptActions=@()
foreach ($a in $actions) {
  $overlap = $false
  foreach ($g in $goals) {
    if ($a.t0 -lt $g.t1 -and $a.t1 -gt $g.t0) { $overlap = $true; break }
  }
  if (-not $overlap) { $keptActions += $a }
}

$final = @()
$final += $goals
$final += $keptActions
$final = $final | Sort-Object t0
$final = Merge-Spans $final 0.10

# debug before cap
$__preCount = $final.Count
$__preTotal = (($final | ForEach-Object { $_.t1 - $_.t0 } | Measure-Object -Sum).Sum)
$__budget   = [math]::Round($duration * $MaxFractionOfRaw,3)
Write-Host ("[debug] pre-cap: n={0}, total={1:F3}s, budget={2:F3}s" -f $__preCount, $__preTotal, $__budget)

$final = Cap-ByFraction $final $duration $MaxFractionOfRaw

# drop tiny spans
$final = $final | Where-Object { ($_.t1 - $_.t0) -ge $MinSpanSec }

# debug after cap
$__postCount = $final.Count
$__postTotal = (($final | ForEach-Object { $_.t1 - $_.t0 } | Measure-Object -Sum).Sum)
Write-Host ("[debug] post-cap: n={0}, total={1:F3}s" -f $__postCount, ($__postTotal))

# debug / triage CSV
$debug = $final | Select-Object @{n="start";e={[math]::Round($_.t0,3)}},
                              @{n="end";e={[math]::Round($_.t1,3)}},
                              @{n="len";e={[math]::Round(($_.t1-$_.t0),3)}},
                              src
$debugPath = Join-Path $OutDir "reel_spans_debug.csv"
$debug | Export-Csv $debugPath -NoTypeInformation
Write-Host "[debug] spans written -> $debugPath"
Write-Host ("[debug] counts: goals={0} actions={1} final={2}" -f $goals.Count,$keptActions.Count,$final.Count)

if ($final.Count -eq 0) { Write-Warning "No spans selected. Check your CSVs."; return }

# build filtergraph to file (avoid long commandlines)
$ci = [Globalization.CultureInfo]::InvariantCulture
$parts  = New-Object System.Collections.Generic.List[string]
$labelsV = New-Object System.Collections.Generic.List[string]
$labelsA = New-Object System.Collections.Generic.List[string]

$i = 0
foreach ($s in $final) {
  $i++
  $start = [string]::Format($ci, "{0:F3}", $s.t0)
  $end   = [string]::Format($ci, "{0:F3}", $s.t1)
  $sv = "v$i"; $sa = "a$i"
  $parts.Add("[0:v]trim=start=$start:end=$end,setpts=PTS-STARTPTS[$sv]")
  $parts.Add("[0:a]atrim=start=$start:end=$end,asetpts=PTS-STARTPTS[$sa]")
  $labelsV.Add("[$sv]"); $labelsA.Add("[$sa]")
}
$concatLine = ($labelsV + $labelsA) -join ''
$concatLine += "concat=n=$i:v=1:a=1[vf][a]"
$filter = ($parts + $concatLine) -join ";"

$filterPath = Join-Path $OutDir 'filter_complex.txt'
Set-Content $filterPath $filter -Encoding ascii
Write-Host ("[debug] filter_complex entries={0}, file={1}" -f $i, $filterPath)

$outPath = Join-Path $OutDir "top_highlights_goals_first.mp4"

# single encode -> smooth, no DTS issues
ffmpeg -y -hide_banner -loglevel error -stats -i $Video `
  -filter_complex_script $filterPath -map "[vf]" -map "[a]" `
  -c:v libx264 -preset veryfast -crf 20 -g 48 -sc_threshold 0 -pix_fmt yuv420p `
  -c:a aac -b:a 160k -movflags +faststart `
  $outPath

Write-Host "[done] -> $outPath"
