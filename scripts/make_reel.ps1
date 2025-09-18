param(
  [Parameter(Mandatory=$true)][string]$Video,
  [ValidateSet("ignore","strict","loose")][string]$GoalMode = "strict",
  [int]$MaxGoals = 6,
  [double]$MinGoalSeparation = 20.0,
  [double]$MaxFractionOfRaw = 0.20,
  [string]$OutDir = ".\out",
  [double]$ActionPad = 0.8
)


function TryParse-Double([object]\, [ref] [double]\) {
  \@{t0=247; t1=251.38} = ""\"" -replace ',', ''
  return [double]::TryParse(\@{t0=247; t1=251.38},
    [Globalization.NumberStyles]::Float,
    [Globalization.CultureInfo]::InvariantCulture,
    [ref]\)
}
] [string]$GoalMode = "strict",
  [int]$MaxActionLen = 7,
  [int]$MaxGoalLen   = 6,
  [float]$MergeGap   = 1.5,
  [float]$GoalPad    = 1.2,
  [float]$ActionPad  = 0.8,
  [float]$ActionPadBefore = 3.5,
  [float]$ActionPadAfter  = 3.5,
  [float]$MaxFractionOfRaw = 0.25,
  [int]$MaxGoals = 6,
  [float]$MinGoalSeparation = 20.0
)

$ErrorActionPreference = "Stop"
if (!(Test-Path $Video)) { throw "Missing input video: $Video" }
New-Item -ItemType Directory -Force -Path $OutDir, ".\scripts" | Out-Null

$hiCSV = Join-Path $OutDir "plays.csv"         # ranked actions from 05_filter_by_motion.py
$grCSV = Join-Path $OutDir "goal_resets.csv"
if (!(Test-Path $hiCSV)) { throw "Missing $hiCSV (run 05_filter_by_motion.py)" }
if (!(Test-Path $grCSV)) { Write-Host "[warn] No $grCSV; proceeding without goals." }

$goalsDir = Join-Path $OutDir "clips_goals"
$actsDir  = Join-Path $OutDir "clips_top"
$logsDir  = Join-Path $OutDir "logs"
New-Item -ItemType Directory -Force -Path $goalsDir,$actsDir,$logsDir | Out-Null
Remove-Item "$goalsDir\*", "$actsDir\*", (Join-Path $OutDir "concat_goals*.txt"), (Join-Path $OutDir "top_highlights_*.mp4") -Recurse -ErrorAction SilentlyContinue

function Read-Spans($csvPath, $preferStartEnd=$true) {
  $rows = Import-Csv $csvPath; if ($rows.Count -eq 0) { return @() }
  $cols = $rows[0].PSObject.Properties.Name
  $startCol = @("start","t0","start_s","begin","clip_start") | ? { $cols -contains $_ } | select -First 1
  $endCol   = @("end","t1","end_s","finish","clip_end")      | ? { $cols -contains $_ } | select -First 1
  $timeCol  = @("time","t","ts","sec","seconds","goal_time","center","mid") | ? { $cols -contains $_ } | select -First 1
  $out=@()
  foreach ($r in $rows) {
    $t0=$null; $t1=$null
    if ($preferStartEnd -and $startCol -and $endCol) {
      $a=0; [double]$b=0
      if ([double]::TryParse("$($r.$startCol)",[ref]$a) -and [double]::TryParse("$($r.$endCol)",[ref]$b) -and $b -gt $a) { $t0=$a; $t1=$b }
    }
    if ($t0 -eq $null -and $timeCol) {
      [double]$c=0
      if ([double]::TryParse("$($r.$timeCol)",[ref]$c)) { $t0=[math]::Max(0,$c-$GoalPad); $t1=$c+$GoalPad }
    }
    if ($t0 -ne $null -and $t1 -ne $null -and $t1 -gt $t0) { $out += [pscustomobject]@{ t0=[double]$t0; t1=[double]$t1 } }
  }
  $out | Where-Object { $_.t1 - $_.t0 -gt 0.05 }
}

function Get-GoalSpans($csvPath, $mode="strict", $minSep=20.0, $maxN=6) {
  if ($mode -eq "ignore" -or -not (Test-Path $csvPath)) { return @() }
  $rows = Import-Csv $csvPath; if ($rows.Count -eq 0) { return @() }
  $cols = $rows[0].PSObject.Properties.Name
  $startCol = @("start","t0","start_s","begin","clip_start") | ? { $cols -contains $_ } | select -First 1
  $endCol   = @("end","t1","end_s","finish","clip_end")      | ? { $cols -contains $_ } | select -First 1
  $timeCol  = @("time","t","ts","sec","seconds","goal_time","center","mid") | ? { $cols -contains $_ } | select -First 1

  $cands=@()
  foreach ($r in $rows) {
    $txt = ("$($r.is_goal) $($r.goal) $($r.event) $($r.label) $($r.type)").ToLower()
    [double]$sd=0; $hasScore = ([double]::TryParse("$($r.score_delta)",[ref]$sd) -and $sd -gt 0)
    $isGoal = if ($mode -eq "strict") { $hasScore -or ($txt -match "goal|scored|score\+") } else { $hasScore -or ($txt -match "goal|scored|score\+|reset|whistle") }
    if (-not $isGoal) { continue }

    if ($startCol -and $endCol) {
      $a=0; [double]$b=0
      if ([double]::TryParse("$($r.$startCol)",[ref]$a) -and [double]::TryParse("$($r.$endCol)",[ref]$b) -and $b -gt $a) {
        $cands += [pscustomobject]@{ t0=$a; t1=$b }; continue
      }
    }
    if ($timeCol) {
      [double]$c=0
      if ([double]::TryParse("$($r.$timeCol)",[ref]$c)) { $cands += [pscustomobject]@{ t0=[math]::Max(0,$c-$GoalPad); t1=$c+$GoalPad } }
    }
  }

  $cands = $cands | Sort-Object t0
  $out=@()
  foreach ($s in $cands) {
    if ($out.Count -eq 0 -or $s.t0 - $out[-1].t1 -ge $minSep) { $out += $s } else { $out[-1].t1 = [math]::Max($out[-1].t1, $s.t1) }
  }
  if ($maxN -gt 0) { $out = $out | Select-Object -First $maxN }
  $out
}

function Merge-Spans($spans, $gap, $maxLen) {
  $spans = $spans | Sort-Object t0
  $merged=@()
  foreach ($s in $spans) {
    if ($merged.Count -eq 0) { $merged += [pscustomobject]@{ t0=$s.t0; t1=$s.t1 }; continue }
    $last = $merged[-1]
    if ($s.t0 -le ($last.t1 + $gap)) { $last.t1 = [math]::Max($last.t1, $s.t1); $merged[-1] = $last }
    else { $merged += [pscustomobject]@{ t0=$s.t0; t1=$s.t1 } }
  }
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
    [double]::Parse($p, [globalization.cultureinfo]::InvariantCulture)
  } catch { return 0 }
}

# 1) read actions (from ranked plays.csv) and optional goals
$rows = Import-Csv (Join-Path $OutDir "plays.csv")
$actionSpans = @()
foreach ($r in $rows) {
  $a=0; [double]$b=0
  if ([double]::TryParse("$($r.start)",[ref]$a) -and [double]::TryParse("$($r.end)",[ref]$b) -and $b -gt $a) {
    $actionSpans += [pscustomobject]@{ t0=[math]::Max(0,$a - $ActionPad); t1=$b + $ActionPad }
  }
}
$goalSpans = Get-GoalSpans $grCSV $GoalMode $MinGoalSeparation $MaxGoals

# 2) merge + cap + dedupe
$goalSpans   = Dedup (Merge-Spans $goalSpans $MergeGap $MaxGoalLen)
$actionSpans = Dedup (Merge-Spans $actionSpans $MergeGap $MaxActionLen)

# 3) remove action windows that touch goal windows (avoid dupes)
$actionSpans = Subtract-Spans $actionSpans $goalSpans 1.5

# 4) total duration limit relative to raw
$rawDur = Get-Duration $Video
$totActions = ($actionSpans | ForEach-Object { $_.t1 - $_.t0 } | Measure-Object -Sum).Sum
$totGoals   = ($goalSpans   | ForEach-Object { $_.t1 - $_.t0 } | Measure-Object -Sum).Sum
$budget = if ($rawDur -gt 0) { [math]::Max(60, $rawDur * $MaxFractionOfRaw) } else { 240 }
if ($totGoals + $totActions -gt $budget) {
  # keep all goals first; trim actions to fit
  $remain = [math]::Max(30, $budget - $totGoals)
  $acc=0.0; $trimmed=@()
foreach ($a in $actionSpans) {
  $t0 = [double]($a.t0)
  $t1 = [double]($a.t1)
  $center = ($t0 + $t1) / 2.0
  $start  = [Math]::Max(0, $center - $ActionPadBefore)
  $end    = [Math]::Min($VideoDuration, $center + $ActionPadAfter)
  $segments += [pscustomobject]@{ kind='action'; start=$start; end=$end }
}
}

[double]$VideoDuration = if ($rawDur -gt 0) { $rawDur } else { [double]::MaxValue }
# Ensure we have a list to append to (faster/safer than += on arrays)
if (-not ($segments -is [System.Collections.Generic.List[object]])) {
  $existing = $segments
  $segments = [System.Collections.Generic.List[object]]::new()
  if ($existing) {
    foreach ($item in $existing) { $null = $segments.Add($item) }
  }
}
foreach ($g in $goalSpans) {
  $t0 = [double]($g.t0)
  $t1 = [double]($g.t1)
  $segments += [pscustomobject]@{ kind='goal'; start=$t0; end=$t1 }
}

# 5) derive action segments centered on span midpoints
# Normalize action spans -> numbers (handles comma thousands + invariant decimals)
$actionSpans = $actionSpans | ForEach-Object {
  $t0s = ($_ | Select-Object -Expand t0) -replace ',', ''
  $t1s = ($_ | Select-Object -Expand t1) -replace ',', ''
  [pscustomobject]@{
    t0 = [double]::Parse($t0s, [Globalization.NumberStyles]::Float -bor [Globalization.NumberStyles]::AllowThousands, [Globalization.CultureInfo]::InvariantCulture)
    t1 = [double]::Parse($t1s, [Globalization.NumberStyles]::Float -bor [Globalization.NumberStyles]::AllowThousands, [Globalization.CultureInfo]::InvariantCulture)
  }
}
foreach ($a in $actionSpans) {
  $t0 = [double]($a.t0)
  $t1 = [double]($a.t1)
  $center = ($t0 + $t1) / 2.0
  $start  = [Math]::Max(0, $center - $ActionPadBefore)
  $end    = [Math]::Min($VideoDuration, $center + $ActionPadAfter)
  $segments += [pscustomobject]@{ kind='action'; start=$start; end=$end }
}

$actionSpans = $segments | Where-Object { $_.kind -eq 'action' } | ForEach-Object {
  [pscustomobject]@{ t0=[double]($_.start); t1=[double]($_.end) }
}

# 6) debug TSV
$tsv = Join-Path $OutDir "segments.tsv"
"kind`tstart`tend`tlen" | Set-Content -Encoding ascii $tsv
$segments | Sort-Object start | % { "{0}`t{1:N2}`t{2:N2}`t{3:N2}" -f $_.kind, $_.start, $_.end, ($_.end-$_.start) } | Add-Content -Encoding ascii $tsv

# 7) cut clips (re-encode for smooth starts)
$iG=0
foreach ($g in $goalSpans) {
  $t0 = [double]($g.t0)
  $t1 = [double]($g.t1)
  $segments += [pscustomobject]@{ kind='goal'; start=$t0; end=$t1 }
}
$iA=0
foreach ($a in $actionSpans) {
  $t0 = [double]($a.t0)
  $t1 = [double]($a.t1)
  $center = ($t0 + $t1) / 2.0
  $start  = [Math]::Max(0, $center - $ActionPadBefore)
  $end    = [Math]::Min($VideoDuration, $center + $ActionPadAfter)
  $segments += [pscustomobject]@{ kind='action'; start=$start; end=$end }
}

# 8) concat lists
$concatGoals = Join-Path $OutDir "concat_goals.txt"
$concatBoth  = Join-Path $OutDir "concat_goals_plus_top.txt"
(Get-ChildItem $goalsDir -Filter *.mp4 -ea SilentlyContinue | Sort-Object Name | % { "file '$($_.FullName)'" }) | Set-Content -Encoding ascii $concatGoals
@(
  Get-ChildItem $goalsDir -Filter *.mp4 -ea SilentlyContinue | Sort-Object Name | % { "file '$($_.FullName)'" }
  Get-ChildItem $actsDir  -Filter *.mp4 -ea SilentlyContinue | Sort-Object Name | % { "file '$($_.FullName)'" }
) -join "`r`n" | Set-Content -Encoding ascii $concatBoth

# 9) final render (consistent GOP + loudness)
$final = Join-Path $OutDir "top_highlights_goals_first.mp4"
ffmpeg -f concat -safe 0 -i $concatBoth -r 24 -g 48 -c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p -c:a aac -ar 48000 -af "loudnorm=I=-16:TP=-1.5:LRA=11" -movflags +faststart -y $final
Write-Host "[done] goals=$iG, actions=$iA -> $final"






