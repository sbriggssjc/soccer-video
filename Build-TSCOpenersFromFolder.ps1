<# Build-TSCOpenersFromFolder.ps1
   Loops over a folder of images named like "Claire Briggs - 14.jpg",
   extracts Name + Number, and calls New-TSCOpeningTitle.ps1 for each.

   USAGE
   -----
   .\Build-TSCOpenersFromFolder.ps1
   (or adjust the $RosterDir / $OutRoot / $OpenerScript paths below)
#>

# --- Your paths ---
$RosterDir    = "C:\Users\scott\OneDrive\Desktop\Personal\Photos\Kids Games\Claire\Photo Circle\Roster"
$OpenerScript = "C:\Users\scott\soccer-video\New-TSCOpeningTitle.ps1"   # the script we built earlier
$OutRoot      = "C:\Users\scott\soccer-video\out\opener"                # where to save all opener MP4s

# --- Prep ---
if (!(Test-Path $RosterDir))    { throw "Roster folder not found: $RosterDir" }
if (!(Test-Path $OpenerScript)) { throw "Opener script not found: $OpenerScript" }
$null = New-Item -ItemType Directory -Force -Path $OutRoot

$regex = '^(?<name>.+?)\s*-\s*(?<num>\d+)\.(jpg|jpeg|png)$'
$manifest = @()

Get-ChildItem -Path $RosterDir -File -Include *.jpg,*.jpeg,*.png | ForEach-Object {
  $m = [regex]::Match($_.Name, $regex, 'IgnoreCase')
  if (-not $m.Success) {
    Write-Warning "Skipping (name pattern not matched): $($_.Name)"
    return
  }

  $name = $m.Groups['name'].Value.Trim()
  $num  = $m.Groups['num'].Value.Trim()

  # Sanitize for file names
  $safeName = ($name -replace '[^\w\s\-]', '_').Trim()

  $outOpener = Join-Path $OutRoot ("TSC_Opener__{0}_{1}.mp4" -f $num, $safeName)

  Write-Host "==> Building opener for $name  #$num"
  & $OpenerScript `
      -PlayerImg  $_.FullName `
      -PlayerName $name `
      -PlayerNo   $num `
      -OutOpener  $outOpener

  $manifest += [pscustomobject]@{
    PlayerName = $name
    PlayerNo   = $num
    ImagePath  = $_.FullName
    OpenerPath = $outOpener
    BuiltAt    = (Get-Date)
  }
}

# Write a quick manifest for reference
$csv = Join-Path $OutRoot "openers_manifest.csv"
$manifest | Export-Csv -NoTypeInformation -Encoding UTF8 $csv
Write-Host "`nAll done. Opener files in: $OutRoot"
Write-Host "Manifest: $csv"

<# Build-TSCOpenersFromFolder.ps1 (patched)
   - Fixes Get-ChildItem filter so files are actually found
   - Accepts any dash character between Name and Number
   - Adds simple discovery stats + per-file echo
#>

# --- Your paths ---
$RosterDir    = "C:\Users\scott\OneDrive\Desktop\Personal\Photos\Kids Games\Claire\Photo Circle\Roster"
$OpenerScript = "C:\Users\scott\soccer-video\New-TSCOpeningTitle.ps1"
$OutRoot      = "C:\Users\scott\soccer-video\out\opener"

# --- Prep ---
if (!(Test-Path $RosterDir))    { throw "Roster folder not found: $RosterDir" }
if (!(Test-Path $OpenerScript)) { throw "Opener script not found: $OpenerScript" }
$null = New-Item -ItemType Directory -Force -Path $OutRoot

# Accept hyphen, en dash, em dash, minus, etc.
# Example matches: "Claire Briggs - 14.jpg" / "Claire Briggs – 14.JPG"
$regex = '^(?<name>.+?)\s*[\p{Pd}]\s*(?<num>\d+)\.(jpg|jpeg|png)$'
$manifest = @()

# IMPORTANT: use wildcard with -Include so it returns files
$files = Get-ChildItem -Path (Join-Path $RosterDir '*') -File -Include *.jpg,*.jpeg,*.png
Write-Host ("Found {0} image(s) in roster folder" -f $files.Count)

foreach ($f in $files) {
  $m = [regex]::Match($f.Name, $regex, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
  if (-not $m.Success) {
    Write-Warning "Skipping (name pattern not matched): $($f.Name)"
    continue
  }

  $name = $m.Groups['name'].Value.Trim()
  $num  = $m.Groups['num'].Value.Trim()

  # Sanitize for file names
  $safeName = ($name -replace '[^\w\s\-]', '_').Trim()
  $outOpener = Join-Path $OutRoot ("TSC_Opener__{0}_{1}.mp4" -f $num, $safeName)

  Write-Host "==> Building opener for $name  #$num"
  & $OpenerScript `
      -PlayerImg  $f.FullName `
      -PlayerName $name `
      -PlayerNo   $num `
      -OutOpener  $outOpener

  $manifest += [pscustomobject]@{
    PlayerName = $name
    PlayerNo   = $num
    ImagePath  = $f.FullName
    OpenerPath = $outOpener
    BuiltAt    = (Get-Date)
  }
}

# Write a quick manifest for reference
$csv = Join-Path $OutRoot "openers_manifest.csv"
$manifest | Export-Csv -NoTypeInformation -Encoding UTF8 $csv
Write-Host "`nAll done. Opener files in: $OutRoot"
Write-Host "Manifest: $csv"
