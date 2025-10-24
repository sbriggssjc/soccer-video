param(
  [string]$Manifest = "C:\\Users\\scott\\soccer-video\\out\\indexes\\atomic_index.csv",
  [string]$OutRoot  = "C:\\Users\\scott\\soccer-video\\out\\atomic_clips",
  [ValidateSet("copy","encode")][string]$Mode = "encode"  # "copy" is GOP-safe only if cuts fall on keyframes
)

$ErrorActionPreference = "Stop"
function Parse-Num($v) {
  if ($null -eq $v) { return $null }
  $s = [string]$v
  $m = [regex]::Matches($s, "[-+]?\d*\.?\d+")
  if ($m.Count -eq 0) { return $null }
  return [double]::Parse($m[0].Value, [System.Globalization.CultureInfo]::InvariantCulture)
}
function Extract-GameKey([string]$clipName) {
  # Expect names like: 123__YYYY-MM-DD__TSC_vs_Whatever__LABEL__t123-t456.mp4
  if ($clipName -match "\d{4}-\d{2}-\d{2}__[^_]+__[^_]+") { return $Matches[0] }
  return "misc"
}
function Cut-Clip([string]$In, [double]$Start, [double]$End, [string]$Out, [string]$Mode) {
  $dur = [Math]::Max(0.01, $End - $Start)
  $args = @("-hide_banner","-loglevel","error","-ss",("{0:n3}" -f $Start),"-i",$In,"-t",("{0:n3}" -f $dur))
  if ($Mode -eq "copy") {
    $args += @("-c","copy","-y",$Out)
  } else {
    $args += @("-c:v","libx264","-preset","veryfast","-crf","18","-c:a","aac","-b:a","128k","-movflags","+faststart","-y",$Out)
  }
  $p = Start-Process -FilePath ffmpeg -ArgumentList $args -NoNewWindow -PassThru -Wait
  if ($p.ExitCode -ne 0) { throw "ffmpeg failed ($($p.ExitCode)) for $Out" }
}

if (!(Test-Path $Manifest)) { throw "Manifest not found: $Manifest" }
$rows = Import-Csv $Manifest

# Try to use t_start_s / t_end_s. If empty, fall back to parsing from clip_name (__t123-t456)
foreach ($r in $rows) {
  $r.t_start_s = Parse-Num $r.t_start_s
  $r.t_end_s   = Parse-Num $r.t_end_s
  if (-not $r.t_start_s -or -not $r.t_end_s) {
    if ($r.clip_name -match "__t([0-9]+(?:\.[0-9]+)?)\s*[-_]\s*t?([0-9]+(?:\.[0-9]+)?)") {
      $r.t_start_s = [double]$Matches[1]
      $r.t_end_s   = [double]$Matches[2]
    }
  }
}

# Canonical key: master + label + start/end (rounded tenths)
$canon = $rows |
  Where-Object { $_.master_path -and $_.clip_name -and $_.t_start_s -ne $null -and $_.t_end_s -ne $null } |
  ForEach-Object {
    $_ | Add-Member -NotePropertyName key -NotePropertyValue (
      ($_.master_path + "|" + ($_.label ?? $_.label_norm ?? "") + "|" +
       ([math]::Round([double]$_.t_start_s,1)) + "-" + ([math]::Round([double]$_.t_end_s,1)))
    ) -Force
    $_
  } |
  Group-Object key | ForEach-Object { $_.Group | Select-Object -First 1 }

$made = 0
foreach ($r in $canon) {
  $master = $r.master_path
  if (!(Test-Path $master)) { Write-Warning "Missing master: $master"; continue }

  # Ensure out dir by game
  $gameKey = Extract-GameKey $r.clip_name
  $outDir = Join-Path $OutRoot $gameKey
  if (!(Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }

  $outPath = Join-Path $outDir $r.clip_name
  try {
    Cut-Clip -In $master -Start ([double]$r.t_start_s) -End ([double]$r.t_end_s) -Out $outPath -Mode $Mode
    $made++
  } catch {
    Write-Warning $_
  }
}
Write-Host "Reclip complete. Wrote $made file(s)."
