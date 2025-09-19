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
  [double]$MaxFractionOfRaw = 0.25,

  [switch]$StrictRecall = $false,
  [switch]$DebugOverlay = $false
)

# ---------- helpers ----------
function Test-DoubleParse([object]$x, [ref]$out) {
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
  [double]$d = 0; Test-DoubleParse $durStr ([ref]$d) | Out-Null
  return [math]::Round($d,3)
}

function Read-Spans([string]$csvPath, [double]$padBefore=0.0, [double]$padAfter=0.0, [string]$source="", [string]$typeLabel="ACTION") {
  $sp=@()
  if (-not (Test-Path $csvPath)) { return $sp }
  $rows = Import-Csv $csvPath
  foreach ($r in $rows) {
    [double]$a=0; [double]$b=0
    if (Test-DoubleParse $r.start ([ref]$a) -and Test-DoubleParse $r.end ([ref]$b) -and $b -gt $a) {
      $t0 = [math]::Max(0,$a - $padBefore)
      $t1 = $b + $padAfter
      if ($t1 -gt $t0) {
        $obj = [pscustomobject]@{ t0=$t0; t1=$t1; src=$source }
        $obj | Add-Member -NotePropertyName type -NotePropertyValue $typeLabel
        $obj | Add-Member -NotePropertyName score -NotePropertyValue 0.0
        $obj | Add-Member -NotePropertyName team -NotePropertyValue "unknown"
        $obj | Add-Member -NotePropertyName source_label -NotePropertyValue $source
        $obj | Add-Member -NotePropertyName reason -NotePropertyValue ""
        $sp += $obj
      }
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

function Get-SpansByFraction($spans, [double]$videoDur, [double]$maxFrac) {
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
      if (Test-DoubleParse $r.start ([ref]$a) -and Test-DoubleParse $r.end ([ref]$b) -and $b -gt $a) {
        Test-DoubleParse $r.score ([ref]$sc) | Out-Null
        $cand += [pscustomobject]@{ t0=$a; t1=$b; score=$sc; src="goal" }
      }
    }
  }

  if ($mode -eq "loose" -and (Test-Path $fgCSV)) {
    $rows = Import-Csv $fgCSV
    foreach ($r in $rows) {
      [double]$a=0; [double]$b=0
      if (Test-DoubleParse $r.start ([ref]$a) -and Test-DoubleParse $r.end ([ref]$b) -and $b -gt $a) {
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
$eventsSelectedCSV = Join-Path $OutDir "events_selected.csv"
$eventsRawCSV = Join-Path $OutDir "events_raw.csv"
$summaryJson = Join-Path $OutDir "review_summary.json"

$usingFused = Test-Path $eventsSelectedCSV

$final = @()
$goals = @()
$keptActions = @()

if ($usingFused) {
  Write-Host "[info] using events_selected.csv"
  $rows = Import-Csv $eventsSelectedCSV
  foreach ($r in $rows) {
    $type = ("" + $r.type).ToUpper()
    if ($type -eq "STOPPAGE") { continue }
    [double]$s = 0; [double]$e = 0; [double]$sc = 0
    if (-not (Test-DoubleParse $r.start ([ref]$s) -and Test-DoubleParse $r.end ([ref]$e))) { continue }
    if ($e -le $s) { continue }
    if ($r.PSObject.Properties.Match('score').Count -gt 0) {
      Test-DoubleParse $r.score ([ref]$sc) | Out-Null
    }
    $obj = [pscustomobject]@{ t0=$s; t1=$e; src=$r.source }
    $obj | Add-Member -NotePropertyName type -NotePropertyValue $type
    $obj | Add-Member -NotePropertyName score -NotePropertyValue $sc
    $obj | Add-Member -NotePropertyName team -NotePropertyValue ($r.team)
    $obj | Add-Member -NotePropertyName source_label -NotePropertyValue ($r.source)
    $obj | Add-Member -NotePropertyName reason -NotePropertyValue ($r.reason)
    $obj | Add-Member -NotePropertyName event_ids -NotePropertyValue ($r.event_ids)
    $final += $obj
  }
  $final = $final | Sort-Object t0, t1
  $goals = $final | Where-Object { $_.type -eq "GOAL" }
  $keptActions = $final | Where-Object { $_.type -ne "GOAL" }
  $__preCount = $final.Count
  $__preTotal = (($final | ForEach-Object { $_.t1 - $_.t0 } | Measure-Object -Sum).Sum)
  $__budget   = [math]::Round($duration * $MaxFractionOfRaw,3)
  Write-Host ("[debug] fused pre-cap: n={0}, total={1:F3}s, budget={2:F3}s" -f $__preCount, $__preTotal, $__budget)
  if ($__preTotal -gt ($__budget + 0.01)) {
    Write-Warning ("Fused selection exceeds requested cap ({0:F3}s > {1:F3}s)" -f $__preTotal, $__budget)
  }
  $__postCount = $__preCount
  $__postTotal = $__preTotal
} else {
  $goals = Get-GoalSpans $OutDir $GoalMode $MinGoalSeparation $MaxGoals
  # pad goals (pre/post) and clamp within video duration
  $goals = $goals | ForEach-Object {
    $s = $_
    [double]$t0 = 0
    [double]$t1 = 0
    Test-DoubleParse $s.t0 ([ref]$t0) | Out-Null
    Test-DoubleParse $s.t1 ([ref]$t1) | Out-Null

    $t0 = [math]::Max(0, $t0 - $GoalPadBefore)
    $t1 = [math]::Min($duration, $t1 + $GoalPadAfter)

    # compute score without using a ternary (PS5.1-safe)
    $sc = 0.0
    if ($s.PSObject.Properties.Match('score').Count -gt 0) {
      [double]$tmp = 0
      if (Test-DoubleParse $s.score ([ref]$tmp)) { $sc = $tmp }
    }
    $obj = [pscustomobject]@{ t0 = $t0; t1 = $t1; src = $s.src; score = $sc }
    $obj | Add-Member -NotePropertyName type -NotePropertyValue "GOAL"
    $obj | Add-Member -NotePropertyName team -NotePropertyValue "us"
    $obj | Add-Member -NotePropertyName source_label -NotePropertyValue $s.src
    $obj | Add-Member -NotePropertyName reason -NotePropertyValue "goal_csv"
    $obj
  }

  $actions = @()
  if (Test-Path $playsCSV) { $actions += Read-Spans $playsCSV $ActionPad $ActionPad "plays" "ACTION" }
  if (Test-Path $hi2CSV)   { $actions += Read-Spans $hi2CSV $ActionPad $ActionPad "hi_filtered" "ACTION" }
  if (Test-Path $hi1CSV)   { $actions += Read-Spans $hi1CSV $ActionPad $ActionPad "hi_shots" "SHOT" }
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

  $final = Get-SpansByFraction $final $duration $MaxFractionOfRaw

  # drop tiny spans
  $final = $final | Where-Object { ($_.t1 - $_.t0) -ge $MinSpanSec }

  # debug after cap
  $__postCount = $final.Count
  $__postTotal = (($final | ForEach-Object { $_.t1 - $_.t0 } | Measure-Object -Sum).Sum)
  Write-Host ("[debug] post-cap: n={0}, total={1:F3}s" -f $__postCount, ($__postTotal))
}

# debug / triage CSV
$debug = $final | Select-Object @{n="start";e={[math]::Round($_.t0,3)}},
                              @{n="end";e={[math]::Round($_.t1,3)}},
                              @{n="len";e={[math]::Round(($_.t1-$_.t0),3)}},
                              @{n="type";e={$_.type}},
                              @{n="score";e={$_.score}},
                              @{n="team";e={$_.team}},
                              @{n="source";e={$_.source_label}},
                              @{n="reason";e={$_.reason}}
$debugPath = Join-Path $OutDir "reel_spans_debug.csv"
$debug | Export-Csv $debugPath -NoTypeInformation
Write-Host "[debug] spans written -> $debugPath"
Write-Host ("[debug] counts: goals={0} actions={1} final={2}" -f $goals.Count,$keptActions.Count,$final.Count)

if ($final.Count -eq 0) { Write-Warning "No spans selected. Check your CSVs."; return }

if ($StrictRecall) {
  if (-not $usingFused) {
    throw "StrictRecall requires events_selected.csv. Run select_events.py first."
  }
  if (-not (Test-Path $summaryJson)) {
    throw "Missing review_summary.json; cannot enforce StrictRecall."
  }

  $summary = Get-Content $summaryJson -Raw | ConvertFrom-Json
  $lookup = @{}
  if (Test-Path $eventsRawCSV) {
    $rows = Import-Csv $eventsRawCSV
    foreach ($row in $rows) {
      $id = "" + $row.event_id
      if (-not [string]::IsNullOrWhiteSpace($id)) {
        $lookup[$id] = $row
      }
    }
  }

  $missRows = @()
  $failMsgs = @()

  $goalTotal = [int]($summary.coverage.goals_total)
  $goalIncluded = [int]($summary.coverage.goals_included)
  if ($goalIncluded -lt $goalTotal) {
    $failMsgs += "goals $goalIncluded/$goalTotal"
    foreach ($id in @($summary.missing.goals)) {
      if (-not $id) { continue }
      if ($lookup.ContainsKey($id)) {
        $row = $lookup[$id]
        [double]$gs = 0; [double]$ge = 0
        Test-DoubleParse $row.start ([ref]$gs) | Out-Null
        Test-DoubleParse $row.end ([ref]$ge) | Out-Null
        $missRows += [pscustomobject]@{
          event_id = $id
          type = $row.type
          start = [math]::Round($gs,3)
          end = [math]::Round($ge,3)
          reason = $row.reason
        }
      } else {
        $missRows += [pscustomobject]@{
          event_id = $id
          type = "GOAL"
          start = ""
          end = ""
          reason = "not found in events_raw"
        }
      }
    }
  }

  $shotTotal = [int]($summary.coverage.shots_total)
  $shotIncluded = [int]($summary.coverage.shots_included)
  if ($shotIncluded -lt $shotTotal) {
    $failMsgs += "shots $shotIncluded/$shotTotal"
    foreach ($id in @($summary.missing.shots)) {
      if (-not $id) { continue }
      if ($lookup.ContainsKey($id)) {
        $row = $lookup[$id]
        [double]$ss = 0; [double]$se = 0
        Test-DoubleParse $row.start ([ref]$ss) | Out-Null
        Test-DoubleParse $row.end ([ref]$se) | Out-Null
        $missRows += [pscustomobject]@{
          event_id = $id
          type = $row.type
          start = [math]::Round($ss,3)
          end = [math]::Round($se,3)
          reason = $row.reason
        }
      } else {
        $missRows += [pscustomobject]@{
          event_id = $id
          type = "SHOT"
          start = ""
          end = ""
          reason = "not found in events_raw"
        }
      }
    }
  }

  if ($failMsgs.Count -gt 0) {
    $missPath = Join-Path $OutDir "strict_recall_misses.csv"
    $missRows | Export-Csv $missPath -NoTypeInformation
    throw "Strict recall failed: " + ($failMsgs -join '; ')
  } else {
    Write-Host ("[StrictRecall] goals {0}/{1} shots {2}/{3}" -f $goalIncluded,$goalTotal,$shotIncluded,$shotTotal)
  }
}

# build filtergraph to file (avoid long commandlines)
$ci = [Globalization.CultureInfo]::InvariantCulture
$parts  = New-Object System.Collections.Generic.List[string]
$labelsV = New-Object System.Collections.Generic.List[string]
$labelsA = New-Object System.Collections.Generic.List[string]

$i = 0
foreach ($s in $final) {
  $i++
  $start = [string]::Format($ci, "{0:F3}", $s.t0)
  $dur   = [string]::Format($ci, "{0:F3}", ($s.t1 - $s.t0))
  $sv = "v$i"; $sa = "a$i"
  $parts.Add("[0:v]trim=start=${start}:duration=${dur},setpts=PTS-STARTPTS[$sv]")
  $parts.Add("[0:a]atrim=start=${start}:duration=${dur},asetpts=PTS-STARTPTS[$sa]")
  $labelsV.Add("[$sv]"); $labelsA.Add("[$sa]")
}
$concatLine = ($labelsV + $labelsA) -join ''
$concatLine += ("concat=n={0}:v=1:a=1[v][a]" -f $i)
$filter = ($parts + $concatLine) -join ";"

$filterPath = Join-Path $OutDir 'filter_complex.txt'
Set-Content $filterPath $filter -Encoding ascii
Write-Host ("[debug] filter_complex entries={0}, file={1}" -f $i, $filterPath)

$outPath = Join-Path $OutDir "top_highlights_goals_first.mp4"

# single encode -> smooth, no DTS issues
ffmpeg -y -hide_banner -loglevel error -stats -i $Video `
  -filter_complex_script $filterPath -map "[v]" -map "[a]" `
  -c:v libx264 -preset veryfast -crf 20 -g 48 -sc_threshold 0 -pix_fmt yuv420p `
  -c:a aac -b:a 160k -movflags +faststart `
  $outPath

Write-Host "[done] -> $outPath"

if ($DebugOverlay) {
  if (-not $usingFused) {
    Write-Warning "DebugOverlay requested but events_selected.csv not found; skipping overlay."
  } else {
    $srtPath = Join-Path $OutDir "events_overlay.srt"
    try {
      & python -u .\scripts\make_overlay.py --events $eventsSelectedCSV --out $srtPath --mode reel | Write-Host
    } catch {
      Write-Warning "Failed to generate overlay SRT: $_"
    }
    if (Test-Path $srtPath) {
      $overlayFull = (Resolve-Path $srtPath).Path.Replace('\\','/')
      $vfArg = "subtitles='${overlayFull}'"
      $debugOut = Join-Path $OutDir "debug_preview.mp4"
      ffmpeg -y -hide_banner -loglevel error -stats -i $outPath -vf $vfArg -c:v libx264 -preset veryfast -crf 20 -c:a copy $debugOut
      Write-Host "[debug] debug_preview with overlay -> $debugOut"
    }
  }
}

function Test-Spans($spans) {
  $bad = @(); $ok = @()
  foreach ($s in $spans) {
    [double]$a = 0; [double]$b = 0
    $s0 = (""+$s.t0) -replace ',', ''
    $s1 = (""+$s.t1) -replace ',', ''
    Test-DoubleParse $s0 ([ref]$a) | Out-Null
    Test-DoubleParse $s1 ([ref]$b) | Out-Null

    $srcVal = $null
    if ($s.PSObject.Properties.Match('src').Count -gt 0) { $srcVal = $s.src }

    if ($b -le $a) {
      $bad += [pscustomobject]@{ t0=$a; t1=$b; src=$srcVal }
    } else {
      $props = [ordered]@{ t0=$a; t1=$b }
      if ($null -ne $srcVal) { $props.src = $srcVal }
      if ($s.PSObject.Properties.Match('score').Count -gt 0) {
        [double]$scoreVal = 0
        if (Test-DoubleParse $s.score ([ref]$scoreVal)) { $props.score = $scoreVal }
      }
      $ok  += [pscustomobject]$props
    }
  }
  if ($bad.Count -gt 0) {
    Write-Warning ("Dropping {0} invalid span(s) (t1<=t0). First few:" -f $bad.Count)
    $bad | Select-Object -First 6 @{n='t0';e={[math]::Round($_.t0,3)}}, @{n='t1';e={[math]::Round($_.t1,3)}}, src |
      Format-Table | Out-String | Write-Host
  }
  return $ok
}

