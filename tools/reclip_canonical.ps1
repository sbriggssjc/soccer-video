param(
  [Parameter(Mandatory=$true)][string]$Manifest,
  [Parameter(Mandatory=$true)][string]$OutRoot,
  [ValidateSet("copy","encode")][string]$Mode = "encode"
)

$ErrorActionPreference = "Stop"
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture

function Parse-Num([object]$v) {
  if ($null -eq $v) { return $null }
  $s = [string]$v
  $m = [regex]::Matches($s, "[-+]?\d*\.?\d+")
  if ($m.Count -eq 0) { return $null }
  return [double]::Parse($m[0].Value, $Invariant)
}

function Extract-GameKey([string]$clipName) {
  if ($clipName -match "\d{4}-\d{2}-\d{2}__[^_]+_vs_[^_]+") { return $Matches[0] }
  return "misc"
}

function Coalesce-Label($r) {
  if ($r.label)      { return $r.label }
  if ($r.label_norm) { return $r.label_norm }
  return ""
}

function Cut-Clip([string]$In, [double]$Start, [double]$End, [string]$Out, [string]$Mode) {
  if (-not (Test-Path $In)) { throw "Master not found: $In" }
  $dur = [Math]::Max(0.01, $End - $Start)

  # Locale-safe formatting (no thousands separators)
  $ss = [string]::Format($Invariant, "{0:0.###}", $Start)
  $tt = [string]::Format($Invariant, "{0:0.###}", $dur)

  $args = @("-hide_banner","-loglevel","error","-ss",$ss,"-i",$In,"-t",$tt)
  if ($Mode -eq "copy") {
    $args += @("-c","copy","-y",$Out)
  } else {
    $args += @("-c:v","libx264","-preset","veryfast","-crf","18","-c:a","aac","-b:a","128k","-movflags","+faststart","-y",$Out)
  }

  $p = Start-Process -FilePath ffmpeg -ArgumentList $args -NoNewWindow -PassThru -Wait
  if ($p.ExitCode -ne 0) { throw "ffmpeg exit $($p.ExitCode) for: $Out" }
}

if (!(Test-Path $Manifest)) { throw "Manifest not found: $Manifest" }
$rows = Import-Csv $Manifest

# Normalize rows and build canonical keys
$norm = @()
foreach ($r in $rows) {
  $t1 = Parse-Num $r.t_start_s
  $t2 = Parse-Num $r.t_end_s
  if (-not $t1 -or -not $t2) {
    if ($r.clip_name -match "__t([0-9]+(?:\.[0-9]+)?)\s*[-_]\s*t?([0-9]+(?:\.[0-9]+)?)") {
      $t1 = [double]$Matches[1]
      $t2 = [double]$Matches[2]
    }
  }

  if ($null -eq $t1 -or $null -eq $t2) { continue }
  if ($t2 -le $t1) { continue }
  if (-not $r.master_path -or -not $r.clip_name) { continue }

  $label = Coalesce-Label $r
  $key = ($r.master_path + "|" + $label + "|" +
          [string]::Format($Invariant, "{0:0.0}", [math]::Round([double]$t1,1)) + "-" +
          [string]::Format($Invariant, "{0:0.0}", [math]::Round([double]$t2,1)))

  $norm += [pscustomobject]@{
    key         = $key
    master_path = $r.master_path
    clip_name   = $r.clip_name
    label       = $label
    t1          = [double]$t1
    t2          = [double]$t2
  }
}

# Deduplicate by key (keep first)
$canon = @()
$seen  = @{}
foreach ($n in $norm) {
  if (-not $seen.ContainsKey($n.key)) {
    $seen[$n.key] = $true
    $canon += $n
  }
}

# Write outputs
$made = 0
foreach ($r in $canon) {
  $gameKey = Extract-GameKey $r.clip_name
  $outDir = Join-Path $OutRoot $gameKey
  if (!(Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }
  $outPath = Join-Path $outDir $r.clip_name
  try {
    Cut-Clip -In $r.master_path -Start $r.t1 -End $r.t2 -Out $outPath -Mode $Mode
    $made++
  } catch {
    Write-Warning $_
  }
}

Write-Host "Reclip complete. Wrote $made file(s)."
