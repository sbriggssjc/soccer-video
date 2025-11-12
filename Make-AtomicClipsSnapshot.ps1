$ErrorActionPreference = "Stop"

$rows = @()

# Collect from cinematic stabilized files (source of truth)
Get-ChildItem ".\out\autoframe_work\cinematic" -Directory -Recurse |
  ForEach-Object {
    $follow = Join-Path $_.FullName 'follow'
    $stab   = Join-Path $follow 'stabilized.mp4'
    if (Test-Path $stab) {
      $links = & fsutil hardlink list $stab 2>$null
      $rel   = $links | ForEach-Object {
        $_ -replace '^[A-Za-z]:\\Users\\scott\\soccer-video\\','' -replace '\\','/'
      }
      $brand = $rel | Where-Object { $_ -like 'out/portrait_reels/branded/*_portrait_BRAND.mp4' } | Select-Object -First 1
      $post  = $rel | Where-Object { $_ -like 'out/portrait_reels/branded/*_portrait_POST.mp4' }  | Select-Object -First 1

      $rows += [pscustomobject]@{
        clip_folder     = (Split-Path $_.FullName -Leaf)
        stabilized_path = ($stab -replace '\\','/')
        brand_path      = $brand
        post_path       = $post
        hardlink_count  = ($rel | Measure-Object).Count
      }
    }
}

# Also record any branded files that aren’t hardlinked back (ideally none)
Get-ChildItem ".\out\portrait_reels\branded" -File -Filter *.mp4 | ForEach-Object {
  $links = & fsutil hardlink list $_.FullName 2>$null
  $hasCine = $links | Where-Object { $_ -match '\\out\\autoframe_work\\cinematic\\.*\\follow\\stabilized\.mp4$' }
  if (-not $hasCine) {
    $rows += [pscustomobject]@{
      clip_folder     = '(BRANDED_ORPHAN)'
      stabilized_path = $null
      brand_path      = ($_.FullName -replace '\\','/')
      post_path       = $null
      hardlink_count  = ($links | Measure-Object).Count
    }
  }
}

$csv  = ".\AtomicClips.Manifest.csv"
$json = ".\AtomicClips.Manifest.json"
$rows | Sort-Object clip_folder | Export-Csv -NoTypeInformation -Encoding UTF8 $csv
$rows | Sort-Object clip_folder | ConvertTo-Json -Depth 6 | Out-File -Encoding UTF8 $json

Write-Host "Wrote:`n  $csv`n  $json" -ForegroundColor Cyan
