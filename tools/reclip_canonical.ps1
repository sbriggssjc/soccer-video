param(
  [Parameter(Mandatory=$true)][string]$Manifest,
  [Parameter(Mandatory=$true)][string]$OutRoot,
  [ValidateSet("copy","encode")][string]$Mode = "encode"
)

$ErrorActionPreference = "Stop"

function Parse-Num([object]$v) {
  if ($null -eq $v) { return $null }
  $s = [string]$v
  $m = [regex]::Matches($s, "[-+]?\d*\.?\d+")
  if ($m.Count -eq 0) { return $null }
  return [double]::Parse($m[0].Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Extract-GameKey([string]$clipName) {
  # expects names like: 123__YYYY-MM-DD__Team_vs_Opponent__LABEL__t123-t456.mp4
  if ($clipName -match "\d{4}-\d{2}-\d{2}__[^_]+_vs_[^_]+") { return $Matches[0] }
  return "misc"
}

function Coalesce-Label($r) {
  if ($r.label) { return $r.label }
  if ($r.label_norm) { return $r.label_norm }
  return ""
}

function Cut-Clip([string]$In, [double]$Start, [double]$End, [string]$Out, [string]$Mode) {
  if (-not (Test-Path $In)) { throw "Master not found: $In" }
  $dur = [Math]::Max(0.01, $End - $Start)
  $args = @("-hide_banner","-loglevel","error","-ss",("{0:n3}" -f $Start),"-i",$In,"-t",("{0:n3}" -f $dur))
  if ($Mode -eq "copy") {
    $args += @("-c","copy","-y",$Out)
  } else {
    $args += @("-c:v","libx264","-preset","veryfast","-crf","18","-c:a","aac","-b:a","128k","-movflags","+faststart","-y",$Out)
  }
  $p = Start-Process -FilePath ffmpeg -ArgumentList $args -NoNewWindow -PassThru -Wait
  if ($p.ExitCode -ne 0) { throw "ffmpeg exit $($p.ExitCode) for: $Out" }
}

# Load manifest
if (!(Test-Path $Manifest)) { throw "Manifest not found: $Manifest" }
$rows = Import-Csv $Manifest

# Normalize & build canonical list
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
  if (-not $r.master_path -or -not $r.clip_name -or $null -eq $t1 -or $null -eq $t2) { continue }

  $label = Coalesce-Label $r
  $key = ($r.master_path + "|" + $label + "|" +
          ([math]::Round([double]$t1,1)) + "-" + ([math]::Round([double]$t2,1)))

  $norm += New-Object psobject -Property ([ordered]@{
    key         = $key
    master_path = $r.master_path
    clip_name   = $r.clip_name
    label       = $label
    t1          = [double]$t1
    t2          = [double]$t2
  })
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
