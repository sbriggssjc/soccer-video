param(
  [string]$Video = ".\out\full_game_stabilized.mp4",

  [ValidateSet("ignore","strict","loose")]
  [string]$GoalMode = "loose",

  [int]$MaxGoals = 99,
  [double]$MinGoalSeparation = 5.0,

  [double]$ActionPad = 0.8,
  [double]$GoalPadBefore = 2.0,
  [double]$GoalPadAfter  = 3.0,

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
      $sp += [pscustomobject]@{ t0=$t0; t1=$t1; src=$source }
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

function Cap-ByFraction($spans, [double]$videoDur, [double]$maxFrac) {
  if ($maxFrac -le 0 -or $maxFrac -ge 1) { return $spans }
  $budget = $videoDur * $maxFrac
  $picked=@(); $sum=0.0
  foreach ($s in ($spans | Sort-Object t0)) {
    $len = $s.t1 - $s.t0
    if ($sum + $len -le $budget) { $picked += $s; $sum += $len } else { break }
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

  # fallback/add forced goals in loose mode
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

  $merged = Merge-Spans $cand $minSep
  $ordered = $merged | Sort-Object -Property @{Expression="score";Descending=$true}, @{Expression="t0";Descending=$false}
  if ($maxN -gt 0) { $ordered = $ordered | Select-Object -First $maxN }
  return $ordered
}

# ---------- main ----------
$OutDir = ".\out"
if (!(Test-Path $Video)) { throw "Missing video: $Video" }
$duration = Get-VideoDurationSec $Video

$playsCSV = Join-Path $OutDir "plays.csv"
$hi1CSV   = Join-Path $OutDir "highlights_shots.csv"
$hi2CSV   = Join-Path $OutDir "highlights_filtered.csv"

$goals = Get-GoalSpans $OutDir $GoalMode $MinGoalSeparation $MaxGoals
# pad goals (pre/post)
$goals = $goals | ForEach-Object {
  [pscustomobject]@{ t0=[math]::Max(0, $_.t0 - $GoalPadBefore); t1=[math]::Min($duration, $_.t1 + $GoalPadAfter); src=$_.src }
}

$actions = @()
if (Test-Path $playsCSV) { $actions += Read-Spans $playsCSV $ActionPad $ActionPad "plays" }
if (Test-Path $hi2CSV)   { $actions += Read-Spans $hi2CSV $ActionPad $ActionPad "hi_filtered" }
if (Test-Path $hi1CSV)   { $actions += Read-Spans $hi1CSV $ActionPad $ActionPad "hi_shots" }
$actions = Merge-Spans $actions 0.25

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
$final = Cap-ByFraction $final $duration $MaxFractionOfRaw

# debug / triage output
$debug = $final | Select-Object @{n="start";e={[math]::Round($_.t0,3)}},
                              @{n="end";e={[math]::Round($_.t1,3)}},
                              @{n="len";e={[math]::Round(($_.t1-$_.t0),3)}},
                              src
$debugPath = Join-Path $OutDir "reel_spans_debug.csv"
$debug | Export-Csv $debugPath -NoTypeInformation
Write-Host "[debug] spans written -> $debugPath"
Write-Host ("[debug] counts: goals={0} actions={1} final={2}" -f $goals.Count,$keptActions.Count,$final.Count)

if ($final.Count -eq 0) { Write-Warning "No spans selected. Check your CSVs."; return }

# build filter_complex (one-pass, smooth)
$ci = [Globalization.CultureInfo]::InvariantCulture
$parts = New-Object System.Collections.Generic.List[string]
$labelsV = New-Object System.Collections.Generic.List[string]
$labelsA = New-Object System.Collections.Generic.List[string]

$i = 0
foreach ($s in $final) {
  $i++
  $start = [string]::Format($ci, "{0:F3}", $s.t0)
  $end   = [string]::Format($ci, "{0:F3}", $s.t1)
  $sv = "v$i"; $sa = "a$i"
  $parts.Add("[0:v]trim=start=$start:end=$end, setpts=PTS-STARTPTS[$sv]")
  $parts.Add("[0:a]atrim=start=$start:end=$end, asetpts=PTS-STARTPTS[$sa]")
  $labelsV.Add("[$sv]"); $labelsA.Add("[$sa]")
}

$concatLine = ($labelsV + $labelsA) -join ''
$concatLine += "concat=n=$i:v=1:a=1[v][a]"
$filter = ($parts + $concatLine) -join ";"

$outPath = Join-Path $OutDir "top_highlights_goals_first.mp4"

ffmpeg -y -hide_banner -loglevel error -stats -i $Video `
  -filter_complex $filter -map "[v]" -map "[a]" `
  -c:v libx264 -preset veryfast -crf 20 -g 48 -sc_threshold 0 -pix_fmt yuv420p `
  -c:a aac -b:a 160k -movflags +faststart `
  $outPath

Write-Host "[done] -> $outPath"

