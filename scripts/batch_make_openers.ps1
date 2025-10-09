param(
  [Parameter(Mandatory=$true)] [string] $RosterCsv,
  [Parameter(Mandatory=$true)] [string] $OutDir
)

# Create output directory
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Import-Csv $RosterCsv | ForEach-Object {
  $row = $_

  # --- Clean & validate fields ---
  $name = $row.PlayerName -as [string]
  if ($null -eq $name) { $name = "" }

  $numS = $row.PlayerNumber -as [string]
  if ($null -eq $numS) { $numS = "" }

  $photo = $row.PlayerPhoto -as [string]
  if ($null -eq $photo) { $photo = "" }

  # Pull out the first run of digits from PlayerNumber (handles "#19", " 19 ", "\#14", etc.)
  $m = [regex]::Match($numS, '\d+')
  if (-not $m.Success) {
    Write-Warning "Skipping row (no numeric PlayerNumber): Name='$name' Number='$numS'"
    return
  }
  $num = [int]$m.Value

  if (-not (Test-Path $photo)) {
    Write-Warning "Skipping '$name' (#$num): Photo not found -> $photo"
    return
  }

  # Safe filename for the output
  $safeName = ($name -replace '[^\w\- ]','').Trim() -replace '\s+','_'
  $outFile  = Join-Path $OutDir ("{0}__{1}.mp4" -f $num, $safeName)

  Write-Host "Rendering opener -> $outFile"

  & "$PSScriptRoot\make_opener.ps1" `
    -PlayerName   $name `
    -PlayerNumber $num `
    -PlayerPhoto  $photo `
    -OutFile      $outFile `
    -OutDir       $OutDir
}
