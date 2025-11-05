param(
  [Parameter(Mandatory=$true)][string]$Csv,
  [Parameter(Mandatory=$true)][string]$OutRoot,
  [string]$Brand = "tsc"
)

$ErrorActionPreference = 'Stop'
$FFMPEG = "ffmpeg"

function Parse-Time([string]$s){
  if([string]::IsNullOrWhiteSpace($s)){ return $null }

  $s = $s.Trim()

  # Accept "t123.45" or "T123.45" -> seconds
  if($s -match '^[tT](?<sec>\d+(\.\d+)?)$'){
    return [double]$Matches['sec']
  }

  # Plain seconds "123.45"
  if($s -match '^\d+(\.\d+)?$'){
    return [double]$s
  }

  # Try TimeSpan (accepts "hh:mm:ss", "mm:ss", "hh:mm:ss.fff")
  try {
    $ts = [TimeSpan]::Parse($s, [System.Globalization.CultureInfo]::InvariantCulture)
    return [double]$ts.TotalSeconds
  } catch {}

  throw "Unrecognized time format: '$s' (use seconds like 156.5, or hh:mm:ss(.fff))."
}

function Secs-To-HHMMSSfff([double]$secs){
  if($secs -lt 0){ $secs = 0 }
  $ts = [TimeSpan]::FromSeconds($secs)
  # ffmpeg accepts hh:mm:ss.mmm
  $hh = [Math]::Floor($ts.TotalHours)
  "{0:00}:{1:00}:{2:00}.{3:000}" -f $hh, $ts.Minutes, $ts.Seconds, $ts.Milliseconds
}

function Sanitize-ForPath([string]$name){
  if([string]::IsNullOrWhiteSpace($name)){ return "" }
  # Replace characters invalid on Windows:  \ / : * ? " < > | and control chars
  $name = $name -replace '[:\\\\\/\*\?"<>\|\x00-\x1F]', '_'
  # collapse spaces
  $name = $name -replace '\s{2,}',' '
  $name.Trim()
}

$rows = Import-Csv -LiteralPath $Csv

$made = 0
$skipped = 0
$errors = 0

foreach($r in $rows){
  if([string]::IsNullOrWhiteSpace($r.master_start) -or [string]::IsNullOrWhiteSpace($r.master_end)){
    $skipped++
    continue
  }

  try {
    $startS = Parse-Time $r.master_start
    $endS   = Parse-Time $r.master_end
    if($endS -le $startS){ throw "end <= start ($($r.master_start) -> $($r.master_end))" }

    $startFmt = Secs-To-HHMMSSfff $startS
    $endFmt   = Secs-To-HHMMSSfff $endS

    $id   = "{0}" -f $r.id
    $g    = $r.game_label
    $tag  = if($r.playtag){ $r.playtag } else { "CLIP" }

    $dstDir = Join-Path $OutRoot $g
    if(!(Test-Path $dstDir)){ New-Item -ItemType Directory -Path $dstDir | Out-Null }

    # Safe filename: no colons, etc.
    $tagSafe = Sanitize-ForPath $tag
    $rangeSafe = ("{0}-{1}" -f $r.master_start, $r.master_end) -replace '[:\\\\\/\*\?"<>\|\x00-\x1F]', '_'  # keep your typed format but safe
    $clip = Join-Path $dstDir ("{0}__{1}__{2}.mp4" -f $id, $tagSafe, $rangeSafe)

    # Try stream copy first
    & $FFMPEG -hide_banner -y -ss $startFmt -to $endFmt -i "$($r.master_path)" -c copy "$clip"
    if($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $clip)){
      # Re-encode fallback
      & $FFMPEG -hide_banner -y -ss $startFmt -to $endFmt -i "$($r.master_path)" `
        -vf "scale=1920:-2:flags=lanczos,setsar=1" -r 60 -c:v libx264 -preset slow -crf 18 `
        -c:a aac -b:a 192k -movflags +faststart "$clip"
      if($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $clip)){
        throw "ffmpeg failed to write clip"
      }
    }

    Write-Host "Wrote $clip"
    $made++
  } catch {
    Write-Warning "Row id=$($r.id) tag='$($r.playtag)': $($_.Exception.Message)"
    $errors++
  }
}

Write-Host ("Done. Created={0} Skipped(no times)={1} Errors={2}" -f $made,$skipped,$errors) -ForegroundColor Cyan
