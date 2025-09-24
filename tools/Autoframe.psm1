function Convert-SciToDecimal {
  param(
    [Parameter(Mandatory = $true)][string]$Expression
  )

  if ([string]::IsNullOrWhiteSpace($Expression)) {
    return $Expression
  }

  $pattern = '([0-9]+(?:\.[0-9]+)?)[eE]\+?(-?[0-9]+)'
  return ([System.Text.RegularExpressions.Regex]::Replace($Expression, $pattern, '($1*pow(10,$2))'))
}

function SubN {
  param(
    [Parameter(Mandatory = $true)][string]$Expression,
    [Parameter(Mandatory = $true)][double]$Fps
  )

  if ([string]::IsNullOrWhiteSpace($Expression)) {
    return $Expression
  }

  $culture = [System.Globalization.CultureInfo]::InvariantCulture
  $fpsString = $Fps.ToString($culture)
  return ([System.Text.RegularExpressions.Regex]::Replace($Expression, '\bn\b', "(t*$fpsString)"))
}

function Escape-Commas-In-Parens {
  param(
    [Parameter(Mandatory = $true)][string]$Text
  )

  if ([string]::IsNullOrEmpty($Text)) {
    return $Text
  }

  $builder = [System.Text.StringBuilder]::new()
  $depth = 0
  foreach ($char in $Text.ToCharArray()) {
    switch ($char) {
      '(' {
        $depth++
        [void]$builder.Append($char)
        continue
      }
      ')' {
        if ($depth -gt 0) { $depth-- }
        [void]$builder.Append($char)
        continue
      }
      ',' {
        if ($depth -eq 1) {
          [void]$builder.Append('\,')
        }
        else {
          [void]$builder.Append(',')
        }
        continue
      }
      default {
        [void]$builder.Append($char)
      }
    }
  }

  return $builder.ToString()
}

function Expand-Clip {
  param(
    [Parameter(Mandatory = $true)][string]$Expression,
    [Parameter(Mandatory = $true)][string]$Min,
    [Parameter(Mandatory = $true)][string]$Max
  )

  $inner = if ([string]::IsNullOrWhiteSpace($Expression)) { '0' } else { $Expression }
  $minExpr = if ([string]::IsNullOrWhiteSpace($Min)) { '0' } else { $Min }
  $maxExpr = if ([string]::IsNullOrWhiteSpace($Max)) { '0' } else { $Max }

  $clip = "clip(($inner),$minExpr,$maxExpr)"
  return Escape-Commas-In-Parens -Text $clip
}

Export-ModuleMember -Function \
  Convert-SciToDecimal, \
  SubN, \
  Escape-Commas-In-Parens, \
  Expand-Clip
