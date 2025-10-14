[CmdletBinding()]
param(
  [string]$Root = ".\out\autoframe_work\cinematic",
  [switch]$WhatIf
)

$pattern = 't(?<lead>\d+)\.(?<mid>\d{3})\.(?<tail>\d{2})'

Get-ChildItem -LiteralPath $Root -Directory -Recurse -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match $pattern } |
  ForEach-Object {
    $original = $_.Name
    $fixed = [System.Text.RegularExpressions.Regex]::Replace(
      $original,
      $pattern,
      {
        param($match)
        $lead = $match.Groups['lead'].Value
        $mid  = $match.Groups['mid'].Value
        $tail = $match.Groups['tail'].Value
        return "t$lead$mid.$tail"
      })

    if ($fixed -and $fixed -ne $original) {
      if ($WhatIf) {
        "Would rename: $original -> $fixed"
      } else {
        Rename-Item -LiteralPath $_.FullName -NewName $fixed
        "Renamed: $original -> $fixed"
      }
    }
  }
