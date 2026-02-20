param(
  [Parameter(Mandatory=$true)]
  [string]$SrcDir,                             # Folder containing raw MP4s
  [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
  [string]$Brand   = "tsc",                    # for inventory tags
  [string]$GameLabel = $null,                  # Optional: override game slug / title
  [switch]$Force
)

# --- Binaries ---
$FFMPEG  = "ffmpeg"
$FFPROBE = "ffprobe"

# --- Helpers ---
function Ensure-Dir([string]$p){ if(!(Test-Path -LiteralPath $p)){ New-Item -ItemType Directory -Path $p | Out-Null } }

function Get-CaptureTime([IO.FileInfo]$fi){
  # 1) Try ffprobe "creation_time" (most reliable for iOS)
  try {
    $json = & $FFPROBE -hide_banner -v quiet -print_format json -show_format -show_streams -- "$($fi.FullName)" | ConvertFrom-Json
    $ct   = $null
    if($json.format.tags.creation_time){ $ct = [DateTime]::Parse($json.format.tags.creation_time) }
    elseif($json.streams){
      $json.streams | % {
        if($_.tags.creation_time -and -not $ct){ $ct = [DateTime]::Parse($_.tags.creation_time) }
      }
    }
    if($ct){ return $ct.ToLocalTime() }
  } catch {}

  # 2) Try filename pattern: 20251105_011404000_iOS.MP4
  $m = [regex]::Match($fi.Name, '^(?<d>\d{8})_(?<t>\d{6})(?<ms>\d{3})?_iOS', 'IgnoreCase')
  if($m.Success){
    $d  = $m.Groups['d'].Value  # yyyymmdd
    $t  = $m.Groups['t'].Value  # hhmmss
    $ms = $m.Groups['ms'].Value
    if(-not $ms){ $ms = '000' }
    $s = "{0}-{1}-{2} {3}:{4}:{5}.{6}" -f $d.Substring(0,4),$d.Substring(4,2),$d.Substring(6,2),$t.Substring(0,2),$t.Substring(2,2),$t.Substring(4,2),$ms
    return [DateTime]::Parse($s)
  }

  # 3) Fallback: filesystem CreationTime
  return $fi.CreationTime
}

function Make-Slug([string]$s){
  if([string]::IsNullOrWhiteSpace($s)){ return $null }
  $s = $s -replace '[^\w\s\-]+',''
  $s = $s -replace '\s+','_'
  return $s
}

# --- Discover source files ---
if(!(Test-Path -LiteralPath $SrcDir)){ throw "SrcDir not found: $SrcDir" }
$all = Get-ChildItem -LiteralPath $SrcDir -File | Where-Object {
  $_.Length -gt 0 -and $_.Extension -match '^\.(mp4|mov|m4v)$'
}
if(!$all){ throw "No video files found in: $SrcDir" }

# Sort clips by capture time
$clips = $all | ForEach-Object {
  [pscustomobject]@{
    File = $_
    T    = Get-CaptureTime $_
  }
} | Sort-Object T, @{Expression={$_.File.Name};Ascending=$true}

Write-Host "Discovered $($clips.Count) clips. First/last:" -ForegroundColor Yellow
Write-Host ("  First:  {0}  @ {1}" -f $clips[0].File.Name, $clips[0].T)
Write-Host ("  Last :  {0}  @ {1}" -f $clips[-1].File.Name, $clips[-1].T)

# --- Build game slug / paths ---
# Default game label from folder name + date of first clip
$baseName = Split-Path -Leaf $SrcDir
$dt0      = $clips[0].T
$autoLabel = "{0:yyyy-MM-dd}__{1}" -f $dt0, (Make-Slug $baseName)
if([string]::IsNullOrWhiteSpace($GameLabel)){ $GameLabel = $autoLabel }

$OutRoot     = Join-Path $RepoRoot "out"
$GameRoot    = Join-Path $OutRoot ("masters\" + $GameLabel)
$CatalogRoot = Join-Path $OutRoot "catalog"
$InvDir      = Join-Path $OutRoot "inventory"

Ensure-Dir $OutRoot
Ensure-Dir $GameRoot
Ensure-Dir $CatalogRoot
Ensure-Dir $InvDir

# --- Concat list (no BOM) ---
$listPath = Join-Path $GameRoot "concat_list.txt"
$listLines = $clips | ForEach-Object { "file '$($_.File.FullName.Replace("'", "''"))'" }

# Ensure no BOM: ffmpeg hates BOM at the start (it becomes "﻿file")
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($listPath, $listLines, $utf8NoBom)

# Optional: quick sanity echo
Write-Host "concat_list.txt preview:" -ForegroundColor DarkGray
$listLines | Select-Object -First 3 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }

# --- Master output path ---
$MasterMp4 = Join-Path $GameRoot ("{0}__master.mp4" -f $GameLabel)
if((Test-Path -LiteralPath $MasterMp4) -and -not $Force){
  Write-Host "Master exists (use -Force to overwrite): $MasterMp4" -ForegroundColor Yellow
} else {
  # Stream copy concat (fast, no re-encode). If the sources differ in parameters, fall back to re-encode.
  $concat = & $FFMPEG -hide_banner -y -f concat -safe 0 -i "$listPath" -c copy "$MasterMp4" 2>&1
  if($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $MasterMp4)){
    Write-Host "Direct concat failed; retrying with normalize re-encode (H.264/AAC, 60fps)..." -ForegroundColor Yellow
    $MasterMp4 = Join-Path $GameRoot ("{0}__master_norm.mp4" -f $GameLabel)
    & $FFMPEG -hide_banner -y -f concat -safe 0 -i "$listPath" `
      -vf "scale=1920:-2:flags=lanczos,setsar=1" -r 60 -c:v libx264 -preset slow -crf 18 `
      -c:a aac -b:a 192k -movflags +faststart "$MasterMp4"
    if($LASTEXITCODE -ne 0){ throw "ffmpeg re-encode failed." }
  }
}

# --- Inspect duration ---
$meta = & $FFPROBE -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 -- "$MasterMp4"
$MasterDuration = [Math]::Round([double]$meta, 3)
Write-Host ("Master ready: {0}  (duration ~ {1}s)" -f $MasterMp4, $MasterDuration) -ForegroundColor Green

# --- CSV template for atomic clips ---
$CatalogDir  = Join-Path $CatalogRoot $GameLabel
Ensure-Dir $CatalogDir
$CsvPath     = Join-Path $CatalogDir "events_selected.csv"

# Column design matches prior runs and your season tooling
$headers = @(
  'id','game_label','brand','master_path','master_start','master_end',
  'playtag','phase','side','formation','notes',
  'tag_right_left','score_impact','player','assist','src_hint'
) -join ','

# Preseed 30 empty rows for quick entry
$rows = 1..30 | ForEach-Object {
  $id = "{0:000}" -f $_
  "$id,$GameLabel,$Brand,$MasterMp4,,,,,,, ,,,,"
} | ForEach-Object { $_ -replace '\s+,',',' }  # trim accidental spaces before commas

Set-Content -LiteralPath $CsvPath -Value @($headers) -Encoding UTF8
Add-Content -LiteralPath $CsvPath -Value $rows

Write-Host ("CSV template for atomic clips: {0}" -f $CsvPath) -ForegroundColor Green

# --- Inventory updates (masters + placeholder for atomic events) ---
$MastersInv = Join-Path $InvDir "masters.csv"
$EventsInv  = Join-Path $InvDir "atomic_events.csv"

if(!(Test-Path -LiteralPath $MastersInv)){
  Set-Content -LiteralPath $MastersInv -Value "date_recorded,game_label,brand,master_path,duration_s,source_dir" -Encoding UTF8
}

# Build the line exactly once
$invLine = "{0:yyyy-MM-dd},{1},{2},{3},{4},{5}" -f $dt0,$GameLabel,$Brand,$MasterMp4,$MasterDuration,$SrcDir

# Simple/robust dedupe: use -contains over file lines (no regex parsing issues)
$hasLine = $false
if(Test-Path -LiteralPath $MastersInv){
  $hasLine = (Get-Content -LiteralPath $MastersInv) -contains $invLine
}
if(-not $hasLine){
  Add-Content -LiteralPath $MastersInv -Value $invLine
}

if(!(Test-Path -LiteralPath $EventsInv)){
  Set-Content -LiteralPath $EventsInv -Value "game_label,id,brand,master_path,master_start,master_end,playtag,phase,side,formation,notes,tag_right_left,score_impact,player,assist" -Encoding UTF8
}

# --- Helper file that your existing clipper can consume directly (concat proof) ---
$RunNotes = Join-Path $CatalogDir "README_run_notes.txt"
$notes = @"
How to use:

1) Open the CSV and set 'master_start'/'master_end' (in seconds) for each row you want.
   - Keep 'id' as a 3-digit sequence (001, 002, …).
   - Fill 'playtag' / 'phase' / 'side' / etc. as needed for overlays & inventory.

2) Generate atomic clips (existing pipeline examples):
   powershell -ExecutionPolicy Bypass -File tools\Build-AtomicClipsFromCsv.ps1 `
     -Csv '$CsvPath' -OutRoot '$RepoRoot\out\atomic_clips' -Brand '$Brand'

3) Add the newly created atomic events to inventory:
   powershell -ExecutionPolicy Bypass -File tools\Append-AtomicCsv-To-Inventory.ps1 `
     -Csv '$CsvPath' -Inventory '$EventsInv'

Master: $MasterMp4
Duration: ~${MasterDuration}s
"@
Set-Content -LiteralPath $RunNotes -Value $notes -Encoding UTF8

Write-Host "Done. Master + CSV + inventory entries are in place." -ForegroundColor Cyan
