# === YOLO log -> stats + CSV (v2: latency stats, streaks, gaps.csv) ===
$LogPath = ".\log.txt"
$OutCsv  = ".\detections.csv"
$GapsCsv = ".\gaps.csv"

if (!(Test-Path $LogPath)) { throw "Log file not found at $LogPath. Create it first: notepad $LogPath (paste your YOLO log and save)." }
$log = Get-Content $LogPath

# Pull start/end from filename …t3028.10-t3059.70.mp4 (best-effort)
$fnameMatchObj = $log | Select-String -Pattern 't([\d\.]+)-t([\d\.]+)\.mp4' -AllMatches | Select-Object -First 1
if ($fnameMatchObj) { $m = $fnameMatchObj.Matches[0]; $start=[double]$m.Groups[1].Value; $end=[double]$m.Groups[2].Value } else { $start=0.0; $end=0.0 }

# Total frames from last "frame X/Total"
$lastFrameObj = $log | Select-String -Pattern 'frame\s+\d+\/(\d+)' | Select-Object -Last 1
$totalFrames  = if ($lastFrameObj) { [int]$lastFrameObj.Matches[0].Groups[1].Value } else { 0 }

# FPS: prefer (total / duration). Fallback to 25 if missing.
$fps = if ($totalFrames -gt 0 -and $end -gt $start) { $totalFrames / ($end - $start) } else { 25.0 }

# Regexes (case-insensitive)
$frameRe = [regex]'(?i)frame\s+(\d+)\/\d+'
$msRe    = [regex]'(\d+(?:\.\d+)?)ms'
$ballRe  = [regex]'(?i)\b(\d+)\s+sports ball\b'

$rows = foreach ($line in $log) {
  $f = $frameRe.Match($line); if (-not $f.Success) { continue }
  $frame = [int]$f.Groups[1].Value
  $ballM = $ballRe.Match($line); $nBall = if ($ballM.Success) { [int]$ballM.Groups[1].Value } else { 0 }
  $msM   = $msRe.Match($line);   $lat   = if ($msM.Success) { [double]$msM.Groups[1].Value } else { [double]::NaN }
  $timeSec = if ($start -gt 0 -and $end -gt $start) { $start + ($frame - 1) / $fps } else { ($frame - 1) / $fps }
  [pscustomobject]@{ Frame=$frame; Detected=($nBall -gt 0); Count=$nBall; LatencyMs=$lat; TimeSec=$timeSec }
}

$detFrames = $rows | Where-Object {$_.Detected}
$nodFrames = $rows | Where-Object {-not $_.Detected}

"Total frames:           $totalFrames"
"Frames with ball:       $($detFrames.Count)"
"Frames w/o detections:  $($nodFrames.Count)"
if ($totalFrames -gt 0) { "Detection ratio:       " + [math]::Round(($detFrames.Count / $totalFrames)*100,2) + "%" }

# Latency stats
$allLat = $rows      | Where-Object { $_.LatencyMs -is [double] -and -not [double]::IsNaN($_.LatencyMs) } | Select-Object -Expand LatencyMs
$detLat = $detFrames | Where-Object { -not [double]::IsNaN($_.LatencyMs) } | Select-Object -Expand LatencyMs
function Median([double[]]$nums){ if(!$nums -or $nums.Count -eq 0){ return [double]::NaN }; $s=$nums|Sort-Object; if($s.Count%2){ $s[[int]([math]::Floor($s.Count/2))] } else { ($s[$s.Count/2-1]+$s[$s.Count/2])/2.0 } }

if ($allLat.Count -gt 0) {
  $mAll = $allLat | Measure-Object -Average -Minimum -Maximum
  "Latency (all frames):  avg {0:n1} ms | p50 {1:n1} | min {2:n1} | max {3:n1}" -f `
    $mAll.Average, (Median $allLat), $mAll.Minimum, $mAll.Maximum
}
if ($detLat.Count -gt 0) {
  $mDet = $detLat | Measure-Object -Average -Minimum -Maximum
  "Latency (detections):  avg {0:n1} ms | p50 {1:n1} | min {2:n1} | max {3:n1}" -f `
    $mDet.Average, (Median $detLat), $mDet.Minimum, $mDet.Maximum
}

# First/last detection
if ($detFrames.Count -gt 0) {
  $first = $detFrames[0]; $last = $detFrames[-1]
  "First detection:       frame {0} @ {1:n2}s" -f $first.Frame, $first.TimeSec
  "Last detection:        frame {0} @ {1:n2}s"  -f $last.Frame,  $last.TimeSec
}

# Longest no-detection gap
$gaps = @(); $run = @()
foreach ($r in $rows) { if (-not $r.Detected) { $run += $r } else { if ($run.Count -gt 0) { $gaps += ,$run; $run = @() } } }
if ($run.Count -gt 0) { $gaps += ,$run }
if ($gaps.Count -gt 0 -and $fps -gt 0) {
  $long = $gaps | Sort-Object Count -Descending | Select-Object -First 1
  "Longest gap:           {0} frames (~{1:n2}s), frames {2}-{3}" -f $long.Count, ($long.Count/$fps), $long[0].Frame, $long[-1].Frame

  # Also show top 5 gaps
  "Top 5 no-detection gaps:"
  $gaps | ForEach-Object {
    [pscustomobject]@{
      StartFrame   = $_[0].Frame
      EndFrame     = $_[-1].Frame
      LengthFrames = $_.Count
      StartTimeSec = $_[0].TimeSec
      DurationSec  = $_.Count / $fps
    }
  } | Sort-Object LengthFrames -Descending | Select-Object -First 5 |
    Format-Table -AutoSize
}

# Write gaps.csv (all gaps)
$gapObjs = @()
if ($gaps.Count -gt 0 -and $fps -gt 0) {
  foreach ($g in $gaps) {
    $gapObjs += [pscustomobject]@{
      StartFrame   = $g[0].Frame
      EndFrame     = $g[-1].Frame
      LengthFrames = $g.Count
      StartTimeSec = $g[0].TimeSec
      DurationSec  = $g.Count / $fps
    }
  }
  $gapObjs | Export-Csv -NoTypeInformation $GapsCsv
  "CSV written: $GapsCsv"
}

# Export detections to CSV
$detFrames | Select-Object Frame,TimeSec,Count,LatencyMs | Export-Csv -NoTypeInformation $OutCsv
"CSV written: $OutCsv"
