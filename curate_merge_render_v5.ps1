# === curate_merge_render_v5.ps1 ===
# One-shot Top-10 curation with exclusions, overlap merge, pre-roll, tighter zoom,
# per-clip stabilization (vidstab), image cleanup, audio offset + fades, and a manual
# tweak for highlight #2 (start earlier, end sooner).

# -------- knobs you might tweak --------
$excludeSlots = @(6, 9, 10)   # 1-based positions in the initial Top-10
$take         = 10

# zoom & polish
$zoom         = 1.48          # tighter (1.0 = none)
$preRoll      = 2.0           # add context before action
$gapJoin      = 0.9           # merge windows if overlap or within this many seconds
$fade         = 0.12          # audio fade at part edges (s)

# audio sync: if AUDIO is ahead of VIDEO, DELAY it slightly (positive value)
# try 0.06 .. 0.12. Negative means “play audio earlier”.
$audioOffset  = 0.08

$crf          = 20
$preset       = 'veryfast'
$ci           = [System.Globalization.CultureInfo]::InvariantCulture

# manual special for the 2nd highlight (after ranking/merge)
$secondStartLead = 2.5        # start this many seconds earlier
$secondEndTrim   = 1.0        # trim this many seconds off the end

# optional OpenCV “follow the ball” pass (requires smart_zoom.py)
$followBall   = $false
$followZoom   = 1.55
$followSmooth = 0.85

function New-HashSet([Type]$T) {
  New-Object "System.Collections.Generic.HashSet``1[[$($T.FullName)]]" ([StringComparer]::OrdinalIgnoreCase)
}
# Treat ...\clips\clip_0148.mp4 and ...\clips_acc\clip_0148.mp4 as the same key
function Get-ClipKey([string]$path) {
  if ($path -match 'clip_(\d{3,4})\.mp4$') { return $matches[1] }
  return (Resolve-Path $path).Path
}

# -------- load ranked rows --------
$all = Import-Csv .\out\smart_top10.csv | ForEach-Object {
  [pscustomobject]@{
    path     = $_.path
    inpoint  = [double]$_.inpoint
    duration = [double]$_.duration
    motion   = [double]$_.motion
    audio    = [double]$_.audio
    score    = [double]$_.score
  }
} | Sort-Object score -Descending
if (-not $all -or $all.Count -eq 0) { throw "No rows in out\smart_top10.csv" }

# -------- base top-10 minus exclusions --------
$baseCount = [Math]::Min($take, $all.Count)
$baseTop   = $all[0..($baseCount-1)]
$keep = New-Object System.Collections.Generic.List[object]
for ($i=1; $i -le $baseCount; $i++) { if (-not $excludeSlots.Contains($i)) { $keep.Add($baseTop[$i-1]) } }

# -------- backfill from remaining rows --------
$pool = @()
if ($all.Count -gt $baseCount) { $pool = $all[$baseCount..($all.Count-1)] }

$used = New-HashSet([string])
foreach ($r in $keep) { [void]$used.Add(('{0}|{1:F2}' -f (Get-ClipKey $r.path), $r.inpoint)) }
foreach ($cand in $pool) {
  if ($keep.Count -ge $take) { break }
  $key = '{0}|{1:F2}' -f (Get-ClipKey $cand.path), $cand.inpoint
  if (-not $used.Contains($key)) { $keep.Add($cand); [void]$used.Add($key) }
}
if ($keep.Count -lt $take) { Write-Host "Only $($keep.Count) base clips after exclusions—continuing." -ForegroundColor Yellow }

# -------- merge overlaps / neighbors per clip key --------
function Merge-Intervals {
  param([object[]]$rows, [double]$gap)
  $rows = $rows | Sort-Object { $_.start }
  $merged = New-Object System.Collections.Generic.List[object]
  $cur = $null
  foreach ($r in $rows) {
    $s = $r.start; $e = $r.end
    if (-not $cur) { $cur = [pscustomobject]@{ start=$s; end=$e; score=$r.score; paths=@($r.path) }; continue }
    $touch = ($s - $cur.end) -le $gap
    $over  = $s -lt $cur.end
    if ($touch -or $over) {
      $cur.end   = [Math]::Max($cur.end, $e)
      $cur.score = [Math]::Max($cur.score, $r.score)
      $cur.paths += $r.path
    } else {
      $merged.Add($cur); $cur = [pscustomobject]@{ start=$s; end=$e; score=$r.score; paths=@($r.path) }
    }
  }
  if ($cur) { $merged.Add($cur) }
  return $merged
}

$grouped = $keep | Group-Object { Get-ClipKey $_.path }
$moments = New-Object System.Collections.Generic.List[object]
foreach ($g in $grouped) {
  $rows = $g.Group | ForEach-Object {
    [pscustomobject]@{
      key   = $g.Name
      path  = (Resolve-Path $_.path).Path
      start = $_.inpoint
      end   = $_.inpoint + $_.duration
      score = $_.score
    }
  }
  $merged = Merge-Intervals -rows $rows -gap $gapJoin
  foreach ($m in $merged) {
    $pref  = $m.paths | Sort-Object { $_ -match '\\clips\\' } -Descending | Select-Object -First 1
    $start = [Math]::Max(0.0, $m.start - $preRoll)
    $moments.Add([pscustomobject]@{
      path  = $pref
      start = $start
      dur   = [Math]::Max(0.1, $m.end - $start)
      score = $m.score
    })
  }
}

# rank and keep top N (still sorted by score desc)
$moments = $moments | Sort-Object score -Descending
if ($moments.Count -gt $take) { $moments = $moments[0..($take-1)] }

# --- manual tweak for HIGHLIGHT #2 (1-based): start earlier and end sooner ---
if ($moments.Count -ge 2) {
  $m = $moments[1]          # 0-based index for 2nd item
  $m.start = [Math]::Max(0.0, $m.start - $secondStartLead)
  $m.dur   = [Math]::Max(0.3, $m.dur + $secondStartLead)  # keep original end
  $m.dur   = [Math]::Max(0.3, $m.dur - $secondEndTrim)    # then trim a bit off the end
  $moments[1] = $m
}

# -------- parts folder --------
$partsDir = Join-Path $PWD 'out\smart_parts'
if (-not (Test-Path $partsDir)) { New-Item -ItemType Directory $partsDir | Out-Null }
Get-ChildItem $partsDir -Filter 'part_*.mp4' -ErrorAction SilentlyContinue | Remove-Item -Force

# Stabilization: build (and cache) vidstab transforms per SOURCE clip
$trfDir = Join-Path $partsDir 'trf'
if (-not (Test-Path $trfDir)) { New-Item -ItemType Directory $trfDir | Out-Null }
$trfFor = @{}  # path -> trf

# Audio filter builder (delay or advance + fades, reset PTS)
function New-AFilter([double]$offset, [string]$fadeIn, [string]$fadeOutStart) {
  if ($offset -ge 0) {
    $ms = [int]([math]::Round($offset * 1000))
    return ("adelay=delays={0}:all=1,asetpts=PTS-STARTPTS,afade=t=in:st=0:d={1}:curve=tri,afade=t=out:st={2}:d={1}:curve=tri,aresample=async=1:first_pts=0" -f $ms, $fadeIn, $fadeOutStart)
  } else {
    $cut = -1.0 * $offset
    $cutStr = [string]::Format($ci, "{0:F3}", $cut)
    return ("atrim=start={0},asetpts=PTS-STARTPTS,afade=t=in:st=0:d={1}:curve=tri,afade=t=out:st={2}:d={1}:curve=tri,apad=pad_dur={0},aresample=async=1:first_pts=0" -f $cutStr, $fadeIn, $fadeOutStart)
  }
}
# Audio filter builder (delay or advance + fades, reset PTS)
# (Renamed from Build-AFilter to New-AFilter to use an approved verb)
$concatLines = @()

for ($i=0; $i -lt $moments.Count; $i++) {
  $m   = $moments[$i]
  $src = $m.path
  $st  = [string]::Format($ci, "{0:F3}", $m.start)
  $du  = [string]::Format($ci, "{0:F3}", $m.dur)

  # build vidstab transforms once per source file
  if (-not $trfFor.ContainsKey($src)) {
    $trf = Join-Path $trfDir ("{0}.trf" -f ([IO.Path]::GetFileNameWithoutExtension($src)))
    if (-not (Test-Path $trf)) {
      & ffmpeg -hide_banner -loglevel warning -y -i $src -vf ("vidstabdetect=shakiness=5:accuracy=15:result='{0}'" -f $trf) -f null NUL
      if ($LASTEXITCODE -ne 0) { Write-Host "vidstabdetect failed on $src — falling back to no stabilization for this source." -ForegroundColor Yellow; $trf = $null }
    }
    $trfFor[$src] = $trf
  }
  $trfUse = $trfFor[$src]

  # Build per-clip video filter chain
  $vf = ''
  if ($trfUse) {
    $vf += ("vidstabtransform=input='{0}':smoothing=30:optzoom=0:crop=black,setpts=PTS-STARTPTS," -f $trfUse)
  } else {
    $vf += "setpts=PTS-STARTPTS,"   # still reset PTS if no vidstab
  }
  $vf += ('scale=ceil(iw*{0}/2)*2:ceil(ih*{0}/2)*2,' -f $zoom)
  $vf += ('crop=floor(in_w/{0}/2)*2:floor(in_h/{0}/2)*2:floor((in_w-in_w/{0})/2):floor((in_h-in_h/{0})/2),' -f $zoom)
  $vf += 'hqdn3d=0.6:1.0:6.0:6.0,unsharp=3:3:1.0:3:3:0.35,eq=contrast=1.07:saturation=1.08:gamma=1.02'

  $fi  = [string]::Format($ci, "{0:F3}", $fade)
  $foS = [string]::Format($ci, "{0:F3}", [Math]::Max(0.0, $m.dur - $fade))
  $af  = Build-AFilter -offset $audioOffset -fadeIn $fi -fadeOutStart $foS

  $part = Join-Path $partsDir ("part_{0:000}.mp4" -f ($i+1))
  & ffmpeg -hide_banner -loglevel warning -y `
    -ss $st -t $du -i $src `
    -vf $vf -c:v libx264 -preset $preset -crf $crf `
    -c:a aac -b:a 160k -af "$af" -movflags +faststart `
    $part
  if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed on `"$src`"" }

  if ($followBall) {
    $tmpVid = [System.IO.Path]::ChangeExtension($part, ".video.mp4")
    $outVid = [System.IO.Path]::ChangeExtension($part, ".follow.mp4")
    & ".\.venv\Scripts\python.exe" .\smart_zoom.py --inp $part --out $tmpVid --zoom $followZoom --smooth $followSmooth
    if ($LASTEXITCODE -ne 0) { throw "smart_zoom.py failed on `"$part`"" }
    & ffmpeg -hide_banner -loglevel warning -y -i $tmpVid -i $part -map 0:v:0 -map 1:a:0 -c copy $outVid
    if ($LASTEXITCODE -ne 0) { throw "Remux failed on `"$part`"" }
    Remove-Item $part, $tmpVid -Force
    Rename-Item $outVid $part
  }

  $concatLines += "file '$part'"
}

# -------- concat parts (stream copy) --------
$concatList = Join-Path $partsDir 'concat.txt'
$concatLines | Set-Content -Encoding ascii $concatList

$final = '.\out\smart10_clean_zoom.mp4'
& ffmpeg -hide_banner -loglevel warning -y -safe 0 -f concat -i $concatList -c copy $final
if ($LASTEXITCODE -ne 0) {
  Write-Host "Stream-copy concat had trouble; re-encoding to eliminate timestamp issues…" -ForegroundColor Yellow
  & ffmpeg -hide_banner -loglevel warning -y -safe 0 -f concat -i $concatList `
    -c:v libx264 -preset $preset -crf $crf -c:a aac -b:a 160k -movflags +faststart $final
  if ($LASTEXITCODE -ne 0) { throw "Final concat failed." }
}

Write-Host "✅ Done -> $final" -ForegroundColor Green
