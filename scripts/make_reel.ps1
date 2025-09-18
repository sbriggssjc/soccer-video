param(
  [string]$Video = ".\out\full_game_stabilized.mp4",
  [string]$OutDir = ".\out",
  [int]$MaxActionLen = 7,
  [int]$MaxGoalLen   = 6,
  [float]$MergeGap   = 1.5,
  [float]$GoalPad    = 1.2,
  [float]$ActionPad  = 0.8,
  [float]$MaxFractionOfRaw = 0.6   # keep final selection under ~60% of raw duration
)

$ErrorActionPreference = "Stop"
if (!(Test-Path $Video)) { throw "Missing input video: $Video" }

$hiCSV = Join-Path $OutDir "highlights.csv"
$grCSV = Join-Path $OutDir "goal_resets.csv"
if (!(Test-Path $hiCSV)) { throw "Missing $hiCSV (run 02_detect_events.py first)" }
if (!(Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

$goalsDir = Join-Path $OutDir "clips_goals"
$actsDir  = Join-Path $OutDir "clips_top"
$logsDir  = Join-Path $OutDir "logs"
New-Item -ItemType Directory -Force -Path $goalsDir,$actsDir,$logsDir | Out-Null
Remove-Item "$goalsDir\*", "$actsDir\*", (Join-Path $OutDir "concat_goals*.txt"), (Join-Path $OutDir "top_highlights_*.mp4") -Recurse -ErrorAction SilentlyContinue

function Read-Spans($csvPath, $preferStartEnd=$true) {
  $rows = Import-Csv $csvPath
  if ($rows.Count -eq 0) { return @() }
  $cols = $rows[0].PSObject.Properties.Name
  $startCol = @("start","t0","start_s","begin","clip_start") | Where-Object { $cols -contains $_ } | Select-Object -First 1
  $endCol   = @("end","t1","end_s","finish","clip_end")      | Where-Object { $cols -contains $_ } | Select-Object -First 1
  $timeCol  = @("time","t","ts","sec","seconds","goal_time","center","mid") | Where-Object { $cols -contains $_ } | Select-Object -First 1

  $out=@()
  foreach ($r in $rows) {
    $t0 = $null; $t1 = $null
    if ($preferStartEnd -and $startCol -and $endCol) {
      [double]$a=0; [double]$b=0
      if ([double]::TryParse("$($r.$startCol)",[ref]$a) -and [double]::TryParse("$($r.$endCol)",[ref]$b) -and $b -gt $a) {
        $t0=$a; $t1=$b
      }
    }
    if ($t0 -eq $null -and $timeCol) {
      [double]$c=0
      if ([double]::TryParse("$($r.$timeCol)",[ref]$c)) {
        $t0=[math]::Max(0,$c-$GoalPad); $t1=$c+$GoalPad
      }
    }
    if ($t0 -ne $null -and $t1 -ne $null) { $out += [pscustomobject]@{ t0=[double]$t0; t1=[double]$t1 } }
  }
  $out | Where-Object { $_.t1 - $_.t0 -gt 0.05 }
}

function Merge-Spans($spans, $gap, $maxLen) {
  $spans = $spans | Sort-Object t0
  $merged=@()
  foreach ($s in $spans) {
    if ($merged.Count -eq 0) { $merged += [pscustomobject]@{ t0=$s.t0; t1=$s.t1 }; continue }
    $last = $merged[-1]
    if ($s.t0 -le ($last.t1 + $gap)) {
      $last.t1 = [math]::Max($last.t1, $s.t1)
      $merged[-1] = $last
    } else {
      $merged += [pscustomobject]@{ t0=$s.t0; t1=$s.t1 }
    }
  }
  # cap maximum length; split long spans
  $out=@()
  foreach ($m in $merged) {
    $len = $m.t1 - $m.t0
    if ($len -le $maxLen) { $out += $m; continue }
    $chunks = [math]::Ceiling($len / $maxLen)
    for ($i=0; $i -lt $chunks; $i++) {
      $c0 = $m.t0 + $i * $maxLen
      $c1 = [math]::Min($m.t1, $c0 + $maxLen)
      if ($c1 -gt $c0) { $out += [pscustomobject]@{ t0=$c0; t1=$c1 } }
    }
  }
  $out
}

function Subtract-Spans($A, $B, $pad=1.5) {
  $keep=@()
  foreach ($a in $A) {
    $overlap = $false
    foreach ($b in $B) {
      if ( ($a.t0 - $pad) -lt ($b.t1 + $pad) -and ($b.t0 - $pad) -lt ($a.t1 + $pad) ) { $overlap=$true; break }
    }
    if (-not $overlap) { $keep += $a }
  }
  $keep
}

function Dedup($spans) {
  $seen=@{}; $out=@()
  foreach ($s in ($spans | Sort-Object t0)) {
    $k = ("{0:F2}-{1:F2}" -f $s.t0, $s.t1)
    if (-not $seen.ContainsKey($k)) { $seen[$k]=$true; $out += $s }
  }
  $out
}

function Get-Duration($path) {
  try {
    $p = & ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $path 2>$null
    [double]::Parse(($p -replace ",","."), [globalization.cultureinfo]::InvariantCulture)
  } catch { return 0 }
}

# 1) read and normalize
$actionSpans = Read-Spans $hiCSV $true | ForEach-Object { [pscustomobject]@{ t0=[math]::Max(0, $_.t0 - $ActionPad); t1=$_.t1 + $ActionPad } }
$goalSpans   = if (Test-Path $grCSV) { Read-Spans $grCSV $true } else { @() }

# 2) merge + cap
$goalSpans   = Merge-Spans $goalSpans $MergeGap $MaxGoalLen
$actionSpans = Merge-Spans $actionSpans $MergeGap $MaxActionLen

# 3) drop action overlapping goals (avoid dupes)
$actionSpans = Subtract-Spans $actionSpans $goalSpans 1.5

# 4) dedupe
$goalSpans   = Dedup $goalSpans
$actionSpans = Dedup $actionSpans

# 5) duration guard (use proper measurement)
$rawDur = Get-Duration $Video
$goalDur   = ($goalSpans   | ForEach-Object { $_.t1 - $_.t0 } | Measure-Object -Sum).Sum
$actionDur = ($actionSpans | ForEach-Object { $_.t1 - $_.t0 } | Measure-Object -Sum).Sum
$totDur = $goalDur + $actionDur

if ($rawDur -gt 0 -and $totDur -gt ($rawDur * $MaxFractionOfRaw)) {
  Write-Host ("[warn] Selected {0:N1}s > {1:P0} of raw ({2:N1}s). Trimming actions." -f $totDur, $MaxFractionOfRaw, $rawDur)
  $budget = [math]::Max(60, ($rawDur * $MaxFractionOfRaw) - $goalDur)
  if ($budget -lt 30) { $budget = 30 }
  $acc=0.0; $trimmed=@()
  foreach ($a in $actionSpans) {
    $len = $a.t1 - $a.t0
    if ($acc + $len -le $budget) { $trimmed += $a; $acc += $len }
    else {
      $need = $budget - $acc
      if ($need -gt 1.0) { $trimmed += [pscustomobject]@{ t0=$a.t0; t1=$a.t0+$need }; $acc += $need }
      break
    }
  }
  $actionSpans = $trimmed
  $actionDur = ($actionSpans | ForEach-Object { $_.t1 - $_.t0 } | Measure-Object -Sum).Sum
  $totDur = $goalDur + $actionDur
}

# 6) debug TSV
$tsv = Join-Path $OutDir "segments.tsv"
"kind`tstart`tend`tlen" | Set-Content -Encoding ascii $tsv
$goalSpans   | ForEach-Object { "{0}`t{1:N2}`t{2:N2}`t{3:N2}" -f "goal", $_.t0, $_.t1, ($_.t1-$_.t0) } | Add-Content -Encoding ascii $tsv
$actionSpans | ForEach-Object { "{0}`t{1:N2}`t{2:N2}`t{3:N2}" -f "action", $_.t0, $_.t1, ($_.t1-$_.t0) } | Add-Content -Encoding ascii $tsv

Write-Host ("[plan] goals={0} ({1:N1}s), actions={2} ({3:N1}s), total={4:N1}s" -f $goalSpans.Count, $goalDur, $actionSpans.Count, $actionDur, $totDur)

# 7) cut (re-encode)
$iG=0; $iA=0
foreach ($g in $goalSpans) {
  $iG++; $dst = Join-Path $goalsDir ("goal_{0:D2}.mp4" -f $iG)
  ffmpeg -ss $($g.t0) -to $($g.t1) -i $Video -r 24 -g 48 -c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p -c:a aac -ar 48000 -movflags +faststart -y $dst | Out-Null
}
foreach ($a in $actionSpans) {
  $iA++; $dst = Join-Path $actsDir ("clip_{0:D4}.mp4" -f $iA)
  ffmpeg -ss $($a.t0) -to $($a.t1) -i $Video -r 24 -g 48 -c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p -c:a aac -ar 48000 -movflags +faststart -y $dst | Out-Null
}

# 8) concat + final
$concatGoals = Join-Path $OutDir "concat_goals.txt"
$concatBoth  = Join-Path $OutDir "concat_goals_plus_top.txt"
(Get-ChildItem $goalsDir -Filter *.mp4 -ea SilentlyContinue | Sort-Object Name | ForEach-Object { "file '$($_.FullName)'" }) | Set-Content -Encoding ascii $concatGoals
@(
  Get-ChildItem $goalsDir -Filter *.mp4 -ea SilentlyContinue | Sort-Object Name | ForEach-Object { "file '$($_.FullName)'" }
  Get-ChildItem $actsDir  -Filter *.mp4 -ea SilentlyContinue | Sort-Object Name | ForEach-Object { "file '$($_.FullName)'" }
) -join "`r`n" | Set-Content -Encoding ascii $concatBoth

$final = Join-Path $OutDir "top_highlights_goals_first.mp4"
ffmpeg -f concat -safe 0 -i $concatBoth -r 24 -c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p -c:a aac -ar 48000 -movflags +faststart -y $final
Write-Host ("[done] goals={0}, actions={1}, total~{2:N1}s -> {3}" -f $iG, $iA, $totDur, $final)
