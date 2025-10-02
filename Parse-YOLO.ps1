# === YOLO log -> stats + CSV (fixed: no $matches bleed) ===
$LogPath = ".\log.txt"
$OutCsv  = ".\detections.csv"

if (!(Test-Path $LogPath)) { throw "Log file not found at $LogPath. Create it first: notepad $LogPath (paste your YOLO log and save)." }

$log = Get-Content $LogPath

# Pull start/end times from filename pattern …t3028.10-t3059.70.mp4 (best-effort)
$fnameMatchObj = $log | Select-String -Pattern 't([\d\.]+)-t([\d\.]+)\.mp4' -AllMatches | Select-Object -First 1
if ($fnameMatchObj) {
  $m = $fnameMatchObj.Matches[0]
  $start = [double]$m.Groups[1].Value
  $end   = [double]$m.Groups[2].Value
} else {
  $start = 0.0; $end = 0.0
}

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
  $f = $frameRe.Match($line)
  if (-not $f.Success) { continue }
  $frame = [int]$f.Groups[1].Value

  # Count "sports ball" using explicit regex Match (no $matches)
  $ballM = $ballRe.Match($line)
  $nBall = if ($ballM.Success) { [int]$ballM.Groups[1].Value } else { 0 }

  $msM   = $msRe.Match($line)
  $lat   = if ($msM.Success) { [double]$msM.Groups[1].Value } else { [double]::NaN }

  # TimeSec calculation
  $timeSec = if ($start -gt 0 -and $end -gt $start) { $start + ($frame - 1) / $fps } else { ($frame - 1) / $fps }

  [pscustomobject]@{
    Frame     = $frame
    Detected  = ($nBall -gt 0)
    Count     = $nBall
    LatencyMs = $lat
    TimeSec   = $timeSec
  }
}

$detFrames = $rows | Where-Object {$_.Detected}
$nodFrames = $rows | Where-Object {-not $_.Detected}

"Total frames:           $totalFrames"
"Frames with ball:       " + ($detFrames.Count)
"Frames w/o detections:  " + ($nodFrames.Count)
if ($totalFrames -gt 0) {
  "Detection ratio:       " + [math]::Round(($detFrames.Count / $totalFrames)*100,2) + "%"
}

# Longest no-detection gap
$gaps = @(); $run = @()
foreach ($r in $rows) { if (-not $r.Detected) { $run += $r } else { if ($run.Count -gt 0) { $gaps += ,$run; $run = @() } } }
if ($run.Count -gt 0) { $gaps += ,$run }
if ($gaps.Count -gt 0 -and $fps -gt 0) {
  $long = $gaps | Sort-Object Count -Descending | Select-Object -First 1
  "Longest gap:           {0} frames (~{1:n2}s), frames {2}-{3}" -f $long.Count, ($long.Count/$fps), $long[0].Frame, $long[-1].Frame
}

# Export detections to CSV
$detFrames | Select-Object Frame,TimeSec,Count,LatencyMs | Export-Csv -NoTypeInformation $OutCsv
"CSV written: $OutCsv"
