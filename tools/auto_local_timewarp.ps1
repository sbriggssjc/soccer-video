# --- CONFIG ---
$Proj = "C:\Users\scott\soccer-video"
$Stem = "004__PRESSURE__t640.00-t663.00"
$Clip = Join-Path $Proj "out\atomic_clips\2025-10-12__TSC_SLSG_FallFestival\$Stem.mp4"
$Work = Join-Path $Proj "out\follow_diag\$Stem\auto_refine"
$BasePlan = Join-Path $Work "plan_sh-1_dx-64_dy-128.jsonl"  # your 99.46% micro-best
$BaseCenterT = 3.56        # miss center (s)
$HalfWidth   = 0.60        # local warp half window (s)

# --- GUARDS ---
if (!(Test-Path $Clip -PathType Leaf))   { throw "Missing clip: $Clip" }
if (!(Test-Path $BasePlan -PathType Leaf)){ throw "Missing base plan: $BasePlan" }
if (!(Test-Path $Work)) { New-Item -ItemType Directory -Force -Path $Work | Out-Null }

# --- HELPERS ---
function Has-Prop($o,[string]$n){ $p=$o.PSObject.Properties; if ($p){ return $p.Match($n).Count -gt 0 } else { return $false } }
function Write-Utf8NoBom([string]$Path, [string[]]$Lines){
  $enc = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllLines($Path,$Lines,$enc)
}
function Hann([double]$t,[double]$c,[double]$w){
  if ($t -lt ($c-$w) -or $t -gt ($c+$w)) { return 0.0 }
  $x = ($t-$c)/$w
  return 0.5*(1.0 + [math]::Cos([math]::PI*$x))
}
function Clamp([double]$v,[double]$lo,[double]$hi){ if($v -lt $lo){$lo} elseif($v -gt $hi){$hi} else {$v} }
function Ms([double]$s){ return [int][math]::Round(1000.0*$s) }  # seconds->ms for filenames

# --- LOAD PLAN ---
$series = New-Object System.Collections.Generic.List[object]
$i=0
Get-Content $BasePlan | ? { $_ -and $_.Trim() } | % {
  $o = $_ | ConvertFrom-Json
  $t = if (Has-Prop $o 't') { [double]$o.t } else { $i/24.0 }
  $i++
  $bx=$null;$by=$null
  if     (Has-Prop $o 'bx_stab') { $bx=[double]$o.bx_stab } elseif (Has-Prop $o 'bx') { $bx=[double]$o.bx }
  if     (Has-Prop $o 'by_stab') { $by=[double]$o.by_stab } elseif (Has-Prop $o 'by') { $by=[double]$o.by }
  $series.Add([pscustomobject]@{ t=$t; bx=$bx; by=$by; raw=$_ })
}
if($series.Count -eq 0){ throw "Empty plan: $BasePlan" }
$Tmin = ($series[0]).t
$Tmax = ($series[-1]).t

function Sample-PlanAtT([double]$tt){
  if ($tt -le $Tmin) { return @{bx=$series[0].bx; by=$series[0].by} }
  if ($tt -ge $Tmax) { return @{bx=$series[-1].bx; by=$series[-1].by} }
  $lo=0; $hi=$series.Count-1
  while ($hi-$lo -gt 1){ $mid=[int](($lo+$hi)/2); if ($series[$mid].t -le $tt) { $lo=$mid } else { $hi=$mid } }
  $a=$series[$lo]; $b=$series[$hi]; $den=[math]::Max(1e-6,$b.t-$a.t); $alpha=($tt-$a.t)/$den
  return @{bx=(1-$alpha)*$a.bx + $alpha*$b.bx; by=(1-$alpha)*$a.by + $alpha*$b.by}
}

function Make-LocalTimeWarpPlan([double]$center,[double]$dt,[int]$dx,[int]$dy){
  $name = [IO.Path]::GetFileNameWithoutExtension($BasePlan)
  $cMs  = Ms $center
  $dtMs = Ms $dt
  $Out  = Join-Path $Work ("{0}__c{1}ms_dt{2}ms_dx{3}_dy{4}.jsonl" -f $name,$cMs,$dtMs,$dx,$dy)
  $lines = New-Object System.Collections.Generic.List[string]
  foreach($row in $series){
    $o = $row.raw | ConvertFrom-Json
    $t = $row.t
    $a = [double](Hann $t $center $HalfWidth)
    $tt = $t + $a*$dt
    $samp = Sample-PlanAtT $tt
    $bx = $samp.bx + $a*$dx
    $by = $samp.by + $a*$dy
    if (Has-Prop $o 'bx_stab') { $o.bx_stab = $bx } elseif (Has-Prop $o 'bx') { $o.bx = $bx } else { $o | Add-Member bx $bx }
    if (Has-Prop $o 'by_stab') { $o.by_stab = $by } elseif (Has-Prop $o 'by') { $o.by = $by } else { $o | Add-Member by $by }
    $lines.Add(($o | ConvertTo-Json -Compress))
  }
  Write-Utf8NoBom -Path $Out -Lines $lines
  return $Out
}

function Make-Overlay([string]$PlanPath){
  $ov = $PlanPath -replace '\.jsonl$','__overlay.jsonl'
  $enc = New-Object System.Text.UTF8Encoding($false)
  $outLines = New-Object System.Collections.Generic.List[string]
  [double]$lastBx=[double]::NaN; [double]$lastBy=[double]::NaN; $seen=$false
  Get-Content $PlanPath | ? { $_ -and $_.Trim() } | % {
    $o = $_ | ConvertFrom-Json
    $bx=$null;$by=$null
    if (Has-Prop $o 'bx_stab') { $bx=$o.bx_stab } elseif (Has-Prop $o 'bx') { $bx=$o.bx }
    if (Has-Prop $o 'by_stab') { $by=$o.by_stab } elseif (Has-Prop $o 'by') { $by=$o.by }
    $bxN = ($bx -ne $null -and -not [double]::IsNaN([double]$bx))
    $byN = ($by -ne $null -and -not [double]::IsNaN([double]$by))
    if (-not $seen) { if ($bxN -and $byN) { $seen=$true; $lastBx=[double]$bx; $lastBy=[double]$by } }
    else {
      if (-not $bxN -and -not [double]::IsNaN($lastBx)) { $bx=$lastBx }
      if (-not $byN -and -not [double]::IsNaN($lastBy)) { $by=$lastBy }
      if ($bx -ne $null -and $by -ne $null -and -not [double]::IsNaN([double]$bx) -and -not [double]::IsNaN([double]$by)) { $lastBx=$bx; $lastBy=$by }
    }
    $o|Add-Member -Force bx $bx; $o|Add-Member -Force by $by
    $outLines.Add(($o | ConvertTo-Json -Compress))
  }
  [System.IO.File]::WriteAllLines($ov,$outLines,$enc)
  $dbg = Join-Path (Split-Path $Clip -Parent) ($Stem + ".__OVERLAY_" + ([IO.Path]::GetFileNameWithoutExtension($PlanPath)) + ".mp4")
  python tools\overlay_debug.py --in "$Clip" --telemetry "$ov" --out "$dbg" | Out-Null
  return $dbg
}

function Score-Coverage([string]$PlanPath){
  $p = & python tools\coverage_check.py --ball "$PlanPath" --w 1920 --h 1080 --crop-w 486 --crop-h 864 --margin 90 --smooth 0.25 2>&1
  $m = ($p | Select-String -Pattern 'coverage:\s+(\d+)\/(\d+)\s+=\s+([\d\.]+)%' -AllMatches).Matches
  if($m.Count -gt 0){ return [pscustomobject]@{ text=$p -join "`n"; coverage=[double]$m[0].Groups[3].Value } else { return [pscustomobject]@{ text=$p -join "`n"; coverage=0.0 } }
}

# --- 1) COARSE SWEEP: center jitter + dt ---
$centers = @(); foreach($off in @(-0.06,-0.04,-0.02,0.0,0.02,0.04,0.06)){ $centers += ($BaseCenterT + $off) }
$dtList = @(); for($dt=-0.15; $dt -le 0.15+1e-9; $dt+=0.01){ $dtList += [math]::Round($dt,3) }

$coarse = New-Object System.Collections.Generic.List[object]
foreach($c in $centers){
  foreach($dt in $dtList){
    $cand = Make-LocalTimeWarpPlan $c $dt 0 0
    $score = Score-Coverage $cand
    $coarse.Add([pscustomobject]@{plan=$cand; center=$c; dt=$dt; coverage=$score.coverage})
  }
}
$bestCoarse = $coarse | Sort-Object coverage -Descending | Select-Object -First 1
Write-Host "=== COARSE BEST ==="
$bestCoarse | Format-Table center,dt,coverage -AutoSize

# --- 2) FINE SWEEP: small dx/dy around best center/dt ---
$dxList = -12..12 | Where-Object { $_ % 2 -eq 0 }   # -12..12 step 2
$dyList = -12..12 | Where-Object { $_ % 2 -eq 0 }

$fine = New-Object System.Collections.Generic.List[object]
foreach($dx in $dxList){
  foreach($dy in $dyList){
    $cand = Make-LocalTimeWarpPlan $bestCoarse.center $bestCoarse.dt $dx $dy
    $score = Score-Coverage $cand
    $fine.Add([pscustomobject]@{plan=$cand; dt=$bestCoarse.dt; center=$bestCoarse.center; dx=$dx; dy=$dy; coverage=$score.coverage})
  }
}
$best = $fine | Sort-Object coverage -Descending | Select-Object -First 1
Write-Host "=== LOCAL TIME-WARP BEST (fine) ==="
$best | Format-Table center,dt,dx,dy,coverage -AutoSize

$dbg = Make-Overlay $best.plan
Write-Host ("Overlay: {0}" -f $dbg)
