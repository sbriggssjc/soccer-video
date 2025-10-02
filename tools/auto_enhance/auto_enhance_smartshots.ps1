<#
Auto-enhance (SMART SHOTS): per-shot reanalysis with conditions detection.

Usage:
  .\tools\auto_enhance\auto_enhance_smartshots.ps1 -In "path\to\input.mp4"
  .\tools\auto_enhance\auto_enhance_smartshots.ps1 -In "path\to\folder" -Recurse

Key Options:
  -SceneThresh   0.10..0.60 (default 0.30)   # higher = fewer cuts
  -MinShotSec    minimum shot duration (default 2.0 sec; merges tiny shots)
  -MaxShots      safety cap (default 180)    # prevent thousands of segments
  -SampleStart   seconds to offset within each shot (default 0.2)
  -SampleDur     seconds to analyze per shot (default 4.0)
  -Crf           10..40 (default 20)
  -Preset        x264 preset (default veryfast)
  -TryWB         attempt whitebalance (on by default; auto-fallback if missing)
  -DryRun        log decisions, don’t encode

Outputs:
  - Per-shot temp files in a temp folder
  - Final joined: <input_basename>_ENH_SMARTSHOTS.mp4
  - CSV log next to final: <input_basename>_ENH_SMARTSHOTS_log.csv

Notes:
  • Requires ffmpeg + ffprobe with 'signalstats'. 'whitebalance' is optional.
  • Keeps broadcast-safe via scale in_range=auto:out_range=tv.
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)]
  [string]$In,

  [double]$SceneThresh = 0.30,
  [double]$MinShotSec  = 2.0,
  [int]$MaxShots       = 180,

  [double]$SampleStart = 0.2,
  [double]$SampleDur   = 4.0,

  [ValidateRange(10,40)]
  [int]$Crf = 20,

  [string]$Preset = "veryfast",
  [switch]$Recurse,
  [switch]$TryWB = $true,
  [switch]$DryRun
)

function Get-VideoFiles($root, $recurse) {
  if (Test-Path $root -PathType Leaf) {
    if ($root -match '\.(mp4|mov|mkv|m4v|avi)$') { ,(Resolve-Path $root).Path } else { @() }
  } elseif (Test-Path $root -PathType Container) {
    $opt = @{}; if ($recurse) { $opt.Recurse = $true }
    Get-ChildItem $root -File @opt -Include *.mp4,*.mov,*.mkv,*.m4v,*.avi | ForEach-Object { $_.FullName }
  } else { Write-Error "Path not found: $root"; @() }
}

function Clamp([double]$v,[double]$lo,[double]$hi){ if($v -lt $lo){return $lo}; if($v -gt $hi){return $hi}; $v }

# -------------- Shared helpers --------------
function Get-DurationSec([string]$inPath){
  $durStr = & ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$inPath"
  if (-not $durStr) { throw "Get-DurationSec: ffprobe failed for $inPath" }
  [double]::Parse($durStr, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Get-Scenes-Portable([string]$inPath, [double]$thresh){
  $tmp = [IO.Path]::GetTempFileName()
  $args = @(
    '-hide_banner','-nostats','-i', $inPath,
    '-filter_complex', ("select='gt(scene,{0})',showinfo" -f ('{0:0.###}' -f $thresh)),
    '-f','null','NUL'
  )
  $stderr = & ffmpeg @args 2>&1 | Tee-Object -FilePath $tmp
  $cuts = New-Object System.Collections.Generic.List[double]
  Select-String -Path $tmp -Pattern 'pts_time:([0-9]+\.[0-9]+|[0-9]+)' | ForEach-Object {
    $t = [double]::Parse($_.Matches[0].Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    if ($t -gt 0) { $cuts.Add($t) }
  }
  Remove-Item $tmp -Force -ErrorAction SilentlyContinue
  $cuts | Sort-Object -Unique
}

function Build-FixedChunks([double]$total, [double]$chunkSec){
  if ($chunkSec -le 0) { $chunkSec = 12.0 }
  $ranges = New-Object System.Collections.Generic.List[pscustomobject]
  $t = 0.0
  while ($t -lt $total) {
    $end = [math]::Min($total, $t + $chunkSec)
    $ranges.Add([pscustomobject]@{ Start=$t; End=$end; Dur=($end-$t) })
    $t = $end
  }
  $ranges
}

# -------------- Scene detection --------------
function Get-Scenes([string]$inPath, [double]$thresh, [double]$minDur, [int]$cap) {
  $total = Get-DurationSec $inPath

  # 1) Try ffprobe lavfi (might fail on Windows quoting)
  $cuts = New-Object System.Collections.Generic.List[double]
  try {
    $cmd = @(
      '-v','error',
      '-show_entries','frame=pkt_pts_time:frame_tags=lavfi.scene_score',
      '-of','csv',
      '-f','lavfi',
      ("movie='{0}',select=gt(scene\,{1})" -f ($inPath -replace "'","''"), ('{0:0.###}' -f $thresh))
    )
    $out = & ffprobe @cmd 2>$null
    if ($LASTEXITCODE -eq 0 -and $out) {
      foreach($line in $out -split "`r?`n"){
        if ($line -match 'frame,\s*([\d\.]+)') {
          $t = [double]::Parse($Matches[1], [System.Globalization.CultureInfo]::InvariantCulture)
          if ($t -gt 0 -and $t -lt $total) { $cuts.Add($t) }
        }
      }
    }
  } catch {}

  # 2) If no cuts, try portable ffmpeg stderr parser
  if ($cuts.Count -eq 0) {
    try {
      $cuts = Get-Scenes-Portable $inPath $thresh
    } catch {}
  }

  # 3) Build ranges from cuts, or fall back to fixed chunks, or single-shot
  $ranges = New-Object System.Collections.Generic.List[pscustomobject]
  if ($cuts.Count -gt 0) {
    $all = New-Object System.Collections.Generic.List[double]
    $all.Add(0.0); ($cuts | Sort-Object -Unique) | ForEach-Object { $all.Add($_) }; $all.Add($total)
    for($i=0; $i -lt ($all.Count-1); $i++){
      $start = $all[$i]; $end = $all[$i+1]; $len = $end - $start
      if ($len -lt $minDur -and $ranges.Count -gt 0) {
        $prev = $ranges[$ranges.Count-1]
        $ranges[$ranges.Count-1] = [pscustomobject]@{ Start=$prev.Start; End=$end; Dur=($end-$prev.Start) }
      } else {
        $ranges.Add([pscustomobject]@{ Start=$start; End=$end; Dur=$len })
      }
    }
  } else {
    # fixed chunks → if that somehow fails, single-shot
    $ranges = Build-FixedChunks $total 12.0
    if (-not $ranges -or $ranges.Count -eq 0) {
      $ranges = @([pscustomobject]@{ Start=0.0; End=$total; Dur=$total })
    }
  }

  # Cap total shots
  if ($ranges.Count -gt $cap) {
    $kept = $ranges[0..($cap-2)]
    $lastStart = $kept[-1].End
    $ranges = [System.Collections.Generic.List[object]]$kept
    $ranges.Add([pscustomobject]@{ Start=$lastStart; End=$total; Dur=($total-$lastStart) })
  }

  $ranges
}

# -------------- Stats parsing --------------
function Parse-StatsText([string[]]$lines) {
  $yavgSum = 0.0; $satSum = 0.0; $frames = 0
  $yminMin = 1.0; $ymaxMax = 0.0

  # A) metadata=print style: lavfi.signalstats.YAVG=0.512 ...
  $rxA = [regex]'lavfi\.signalstats\.YAVG=(?<yavg>[\d\.]+).*?lavfi\.signalstats\.YMIN=(?<ymin>[\d\.]+).*?lavfi\.signalstats\.YMAX=(?<ymax>[\d\.]+).*?lavfi\.signalstats\.SATAVG=(?<satavg>[\d\.]+)'

  # B) showinfo style: ... YAVG:0.512 YMIN:0.043 YMAX:0.982 ... SATAVG:0.38 ...
  $rxB = [regex]'YAVG:(?<yavg>[\d\.]+).*?YMIN:(?<ymin>[\d\.]+).*?YMAX:(?<ymax>[\d\.]+).*?SATAVG:(?<satavg>[\d\.]+)'

  foreach ($line in $lines) {
    $m = $rxA.Match($line)
    if (-not $m.Success) { $m = $rxB.Match($line) }
    if ($m.Success) {
      $frames++
      $y    = [double]$m.Groups['yavg'].Value
      $ymin = [double]$m.Groups['ymin'].Value
      $ymax = [double]$m.Groups['ymax'].Value
      $sat  = [double]$m.Groups['satavg'].Value

      $yavgSum += $y; $satSum += $sat
      if ($ymin -lt $yminMin) { $yminMin = $ymin }
      if ($ymax -gt $ymaxMax) { $ymaxMax = $ymax }
    }
  }

  if ($frames -eq 0) { throw "No signalstats lines parsed from stderr." }

  $yavg = $yavgSum / $frames
  $sat  = $satSum  / $frames

  [pscustomobject]@{
    YAVG=$yavg; SAT=$sat; YMIN=$yminMin; YMAX=$ymaxMax; YRNG=($ymaxMax-$yminMin); FRAMES=$frames
  }
}

# -------------- Conditions detection --------------
function Detect-Conditions([pscustomobject]$s) {
  # Heuristics (empirical):
  # Night: YAVG < 0.28 and YRNG < 0.55
  # Overcast: 0.28<=YAVG<=0.48 and SAT < 0.35
  # Daylight: YAVG 0.48..0.70 and SAT 0.35..0.60
  # Backlit: YAVG mid (0.40..0.60) but YMIN << 0.10 and YRNG > 0.75 (deep shadows + hot sky)
  # Indoor: YAVG mid (0.40..0.65) and SAT 0.30..0.55, but weaker YRNG < 0.55
  $cond = "generic"
  if     ($s.YAVG -lt 0.28 -and $s.YRNG -lt 0.55) { $cond = "night" }
  elseif ($s.YAVG -ge 0.28 -and $s.YAVG -le 0.48 -and $s.SAT -lt 0.35) { $cond = "overcast" }
  elseif ($s.YAVG -ge 0.48 -and $s.YAVG -le 0.70 -and $s.SAT -ge 0.35 -and $s.SAT -le 0.60) { $cond = "daylight" }
  elseif ($s.YAVG -ge 0.40 -and $s.YAVG -le 0.60 -and $s.YMIN -lt 0.10 -and $s.YRNG -gt 0.75) { $cond = "backlit" }
  elseif ($s.YAVG -ge 0.40 -and $s.YAVG -le 0.65 -and $s.SAT -ge 0.30 -and $s.SAT -le 0.55 -and $s.YRNG -lt 0.55) { $cond = "indoor" }
  $cond
}

# -------------- EQ decision --------------
function Compute-EQ([double]$yavg,[double]$yrng,[double]$sat,[string]$cond) {
  # Baseline neutral targets
  $targetY  = 0.50
  $targetRg = 0.72
  $targetSa = 0.45

  # Contrast: 1.0 + Clamp((targetRg - yrng) * 0.65, -0.20, 0.22)
  $contrast = 1.0 + (Clamp ( ($targetRg - $yrng) * 0.65 ) -0.20 0.22)
  $contrast = (Clamp $contrast 0.85 1.22)

  # Gamma:
  $gamma = 1.0
  if ($yavg -lt 0.45) {
    # 1.0 + Clamp((0.45 - yavg) * 0.35, 0.00, 0.14)
    $gamma = 1.0 + (Clamp ( (0.45 - $yavg) * 0.35 ) 0.00 0.14)
  } elseif ($yavg -gt 0.62) {
    # 1.0 - Clamp((yavg - 0.62) * 0.22, 0.00, 0.06)
    $gamma = 1.0 - (Clamp ( ($yavg - 0.62) * 0.22 ) 0.00 0.06)
  }
  $gamma = (Clamp $gamma 0.94 1.14)

  # Brightness: Clamp((targetY - yavg) * 0.08, -0.035, 0.035)
  $brightness = (Clamp ( ($targetY - $yavg) * 0.08 ) -0.035 0.035)

  # Saturation gain: 1.0 + Clamp((targetSa - sat) * 0.9, -0.18, 0.22)
  $satGain = 1.0 + (Clamp ( ($targetSa - $sat) * 0.9 ) -0.18 0.22)
  $satGain = (Clamp $satGain 0.82 1.22)

  # Condition tweaks
  switch ($cond) {
    "night"    { $gamma += 0.05; $brightness += 0.01; $contrast -= 0.03; $satGain -= 0.05 }
    "overcast" { $contrast += 0.03; $satGain += 0.06 }
    "daylight" { $contrast += 0.00; $satGain += 0.02 }
    "backlit"  { $gamma += 0.03; $contrast -= 0.02 }
    "indoor"   { $contrast += 0.02; $satGain -= 0.02 }
    default    { }
  }

  # Final clamp + rounding calls must also wrap function invocation:
  [pscustomobject]@{
    contrast   = [math]::Round((Clamp $contrast   0.85 1.24), 3)
    gamma      = [math]::Round((Clamp $gamma      0.92 1.16), 3)
    brightness = [math]::Round((Clamp $brightness -0.04 0.04), 3)
    saturation = [math]::Round((Clamp $satGain    0.80 1.24), 3)
  }
}

# -------------- Filters --------------
function Build-Filter([pscustomobject]$eq, [bool]$wb) {
  $parts = New-Object System.Collections.Generic.List[string]
  $parts.Add('scale=in_range=auto:out_range=tv')
  if ($wb) { $parts.Add('whitebalance=rw=0.96:gw=1.00:bw=1.04:wp=0.9') }  # gentle
  $parts.Add(("eq=contrast={0}:saturation={1}:gamma={2}:brightness={3}" -f `
    $eq.contrast, $eq.saturation, $eq.gamma, $eq.brightness))
  $parts.Add('format=yuv420p')
  ($parts -join ',')
}

# -------------- Sampling one shot --------------
function Sample-Shot([string]$inPath,[double]$start,[double]$dur){
  # Parse signalstats from stderr. We add showinfo so ffmpeg prints per-frame metadata.
  $args = @(
    '-hide_banner','-v','verbose','-ss',('{0:0.###}' -f $start),'-t',('{0:0.###}' -f $dur),
    '-i',$inPath,
    '-vf','scale=in_range=auto:out_range=tv,signalstats,showinfo',
    '-f','null','NUL'
  )
  $stderr = & ffmpeg @args 2>&1
  if ($LASTEXITCODE -ne 0) { throw "signalstats sampling failed ($start..$([math]::Round($start+$dur,3)))" }

  $lines = $stderr -split "`r?`n"
  # Keep only the noisy lines to speed parsing (optional)
  $cand = $lines | Where-Object { $_ -match 'YAVG:|SATAVG:' }
  if (-not $cand -or $cand.Count -eq 0) { $cand = $lines }

  Parse-StatsText $cand
}

# -------------- Encode one shot --------------
function Encode-Shot([string]$inPath,[double]$start,[double]$dur,[string]$filter,[string]$preset,[int]$crf,[string]$tmpDir,[switch]$dry,[switch]$wbInFilter){
  $name = [System.IO.Path]::GetFileNameWithoutExtension($inPath)
  $seg  = Join-Path $tmpDir ("{0}_seg_{1:0.###}-{2:0.###}.mp4" -f $name,$start,($start+$dur))
  $args = @(
    '-hide_banner','-y',
    '-ss',('{0:0.###}' -f $start),'-t',('{0:0.###}' -f $dur),
    '-i',$inPath,
    '-map','0:v:0','-map','0:a:0?',
    '-vf',$filter,
    '-c:v','libx264','-preset',$preset,'-crf',$crf,
    '-c:a','aac','-b:a','192k',
    $seg
  )
  Write-Host "  [ENC] $([IO.Path]::GetFileName($seg))"
  if ($dry) { return $seg }
  $p = Start-Process -FilePath 'ffmpeg' -ArgumentList $args -NoNewWindow -PassThru -Wait
  if ($p.ExitCode -ne 0 -and $wbInFilter) {
    Write-Warning "WB may be unsupported; retrying without WB for this shot…"
    $fallback = ($filter -replace '(^|,)whitebalance=[^,]+,?', '$1').Trim(',')
    $args[$args.IndexOf('-vf')+1] = $fallback
    $p2 = Start-Process -FilePath 'ffmpeg' -ArgumentList $args -NoNewWindow -PassThru -Wait
    if ($p2.ExitCode -ne 0) { throw "FFmpeg failed for shot ($start..$($start+$dur))" }
  } elseif ($p.ExitCode -ne 0) {
    throw "FFmpeg failed for shot ($start..$($start+$dur))"
  }
  $seg
}

# -------------- Join segments --------------
function Concat-Segments([string[]]$paths,[string]$outPath){
  if (-not $paths -or $paths.Count -eq 0) { throw "Concat-Segments: no segments to join." }
  if ($paths.Count -eq 1) {
    # If there's only one segment, just move/rename it to final to avoid concat fragility.
    Copy-Item -LiteralPath $paths[0] -Destination $outPath -Force
    return
  }
  $list = [System.IO.Path]::GetTempFileName()
  $listUtf8 = [System.IO.Path]::ChangeExtension($list,'.txt')
  Remove-Item $list -Force -ErrorAction SilentlyContinue
  $paths | ForEach-Object {
    "file '$($_.Replace("'", "''"))'"
  } | Set-Content -LiteralPath $listUtf8 -Encoding ASCII
  $args = @('-hide_banner','-y','-f','concat','-safe','0','-i',$listUtf8,'-c','copy',$outPath)
  $p = Start-Process -FilePath 'ffmpeg' -ArgumentList $args -NoNewWindow -PassThru -Wait
  if ($p.ExitCode -ne 0) {
    Write-Warning "Stream copy concat failed; re-encoding mux…"
    $args2 = @('-hide_banner','-y','-f','concat','-safe','0','-i',$listUtf8,'-c:v','libx264','-preset','veryfast','-crf','20','-c:a','aac','-b:a','192k',$outPath)
    $p2 = Start-Process -FilePath 'ffmpeg' -ArgumentList $args2 -NoNewWindow -PassThru -Wait
    if ($p2.ExitCode -ne 0) { throw "Concat failed (even with re-encode)." }
  }
  Remove-Item $listUtf8 -Force -ErrorAction SilentlyContinue
}

# -------------- Main per-file --------------
function Process-File([string]$inPath,[double]$sceneThr,[double]$minShot,[int]$cap,[double]$ss,[double]$sd,[string]$preset,[int]$crf,[switch]$tryWB,[switch]$dry){
  $dir  = Split-Path $inPath -Parent
  $base = [System.IO.Path]::GetFileNameWithoutExtension($inPath)
  $final = Join-Path $dir ($base + "_ENH_SMARTSHOTS.mp4")
  $log   = Join-Path $dir ($base + "_ENH_SMARTSHOTS_log.csv")

  Write-Host "`n=== SMART SHOTS: $inPath"
  $shots = Get-Scenes -inPath $inPath -thresh $sceneThr -minDur $minShot -cap $cap
  if (-not $shots -or $shots.Count -eq 0) {
    $total = Get-DurationSec $inPath
    $shots = @([pscustomobject]@{ Start=0.0; End=$total; Dur=$total })
  }
  # Report the number of detected shots so the console always shows a count.
  Write-Host ("Detected {0} shot{1}." -f $shots.Count, $(if($shots.Count -eq 1){''}else{'s'}))

  # Temp dir
  $tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ("smartshots_" + ([System.IO.Path]::GetRandomFileName()))
  New-Item -ItemType Directory -Path $tmpDir | Out-Null

  # CSV log
  "shot_idx,start,end,duration,yavg,yrng,sat,cond,contrast,gamma,brightness,saturation,filter" | Set-Content -LiteralPath $log -Encoding UTF8

  $segments = New-Object System.Collections.Generic.List[string]
  for ($i=0; $i -lt $shots.Count; $i++){
    $s = $shots[$i]
    $sSampleStart = $s.Start + [math]::Min([math]::Max($ss,0.0), [math]::Max($s.Dur - 0.2, 0.0))
    $sSampleDur   = [math]::Min($sd, [math]::Max($s.End - $sSampleStart, 0.2))

    Write-Host ("[SHOT #{0}] {1:0.###} → {2:0.###}  (dur {3:0.###})" -f $i,$s.Start,$s.End,$s.Dur)

    try {
      $stats = Sample-Shot -inPath $inPath -start $sSampleStart -dur $sSampleDur
      $cond  = Detect-Conditions $stats
      $eq    = Compute-EQ -yavg $stats.YAVG -yrng $stats.YRNG -sat $stats.SAT -cond $cond
      $filter = Build-Filter -eq $eq -wb:$tryWB

      # Log
      # Ensure any quotes in the filter string are CSV-safe
      $filterCsv = $filter -replace '"','""'

      # Build a clean CSV row (single-quoted format string, quotes only around the filter column)
      $line = '{0},{1:0.###},{2:0.###},{3:0.###},{4:0.###},{5:0.###},{6:0.###},{7},{8},{9},{10},{11},"{12}"' -f `
        $i, $s.Start, $s.End, $s.Dur, $stats.YAVG, $stats.YRNG, $stats.SAT, $cond, `
        $eq.contrast, $eq.gamma, $eq.brightness, $eq.saturation, $filterCsv
      Add-Content -LiteralPath $log -Value $line -Encoding UTF8

      $seg = Encode-Shot -inPath $inPath -start $s.Start -dur $s.Dur -filter $filter -preset $preset -crf $crf -tmpDir $tmpDir -dry:$dry -wbInFilter:$tryWB
      if ($seg) { $segments.Add($seg) }
    } catch {
      Write-Warning "Shot #$i failed: $_"
    }
  }

  if (-not $dry) {
    if ($segments.Count -eq 0) {
      Write-Warning "No segments encoded; falling back to whole-video smart pass."
      # Analyze mid-video for EQ
      $total = Get-DurationSec $inPath
      $midStart = [math]::Max(0.0, ($total/2.0) - 2.0)
      $eqStats = Sample-Shot -inPath $inPath -start $midStart -dur ([math]::Min(4.0, $total - $midStart))
      $cond    = Detect-Conditions $eqStats
      $eq      = Compute-EQ -yavg $eqStats.YAVG -yrng $eqStats.YRNG -sat $eqStats.SAT -cond $cond
      $filter1 = Build-Filter -eq $eq -wb:$false  # WB off (your build lacks it)

      $args = @(
        '-hide_banner','-y',
        '-i',$inPath,
        '-map','0:v:0','-map','0:a:0?',
        '-vf', $filter1,
        '-c:v','libx264','-preset',$preset,'-crf',$crf,
        '-c:a','aac','-b:a','192k',
        $final
      )
      $p = Start-Process -FilePath 'ffmpeg' -ArgumentList $args -NoNewWindow -PassThru -Wait
      if ($p.ExitCode -ne 0) { throw "Whole-video fallback encode failed." }
      try { Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue } catch {}
      Write-Host "Final: $final"
      Write-Host "Log  : $log"
      return $final
    }
    Concat-Segments -paths $segments.ToArray() -outPath $final
  }

  # Cleanup temp dir
  if (-not $dry) { try { Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue } catch {} }

  Write-Host "Final: $final"
  Write-Host "Log  : $log"
  return $final
}

# -------------- Orchestrate --------------
$files = Get-VideoFiles -root $In -recurse:$Recurse
if (-not $files.Count) { Write-Error "No input videos found."; exit 1 }

foreach($f in $files){
  try {
    Process-File -inPath $f -sceneThr $SceneThresh -minShot $MinShotSec -cap $MaxShots `
      -ss $SampleStart -sd $SampleDur -preset $Preset -crf $Crf -tryWB:$TryWB -dry:$DryRun | Out-Null
  } catch {
    Write-Warning $_
  }
}
