function Convert-SciToDecimal {
  param(
    [Parameter(Mandatory = $true)][string]$Expression
  )

  if ([string]::IsNullOrWhiteSpace($Expression)) {
    return $Expression
  }

  $pattern = '([+-]?\d+(?:\.\d+)?)[eE]([+-]?\d+)'
  $culture = [System.Globalization.CultureInfo]::InvariantCulture
  $format = '0.###############################'

  return [System.Text.RegularExpressions.Regex]::Replace(
    $Expression,
    $pattern,
    {
      param($match)
      try {
        $value = [decimal]::Parse($match.Value, [System.Globalization.NumberStyles]::Float, $culture)
      }
      catch {
        $value = [double]::Parse($match.Value, [System.Globalization.NumberStyles]::Float, $culture)
      }
      return $value.ToString($format, $culture)
    }
  )
}

function Sanitize-Expr {
  param(
    [Parameter(Mandatory = $true)][string]$Expression,
    [Nullable[double]]$Fps
  )

  if ([string]::IsNullOrWhiteSpace($Expression)) {
    return $Expression
  }

  $expr = $Expression -replace '\,', ','
  $expr = Convert-SciToDecimal -Expression $expr
  $expr = ($expr -replace '\s+', '')

  if ($PSBoundParameters.ContainsKey('Fps') -and $null -ne $Fps) {
    $expr = SubN -Expression $expr -Fps $Fps
  }

  return $expr
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

function _Split-TopLevel {
  param(
    [Parameter(Mandatory = $true)][string]$Text,
    [Parameter(Mandatory = $true)][char]$Separator
  )

  $parts = New-Object System.Collections.Generic.List[string]
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
    }

    if ($char -eq $Separator -and $depth -eq 0) {
      $parts.Add($builder.ToString()) | Out-Null
      $builder.Length = 0
      continue
    }

    [void]$builder.Append($char)
  }

  $parts.Add($builder.ToString()) | Out-Null
  return $parts.ToArray()
}

function _Parse-ClipInvocation {
  param(
    [Parameter(Mandatory = $true)][string]$Text,
    [Parameter(Mandatory = $true)][int]$OpenParenIndex
  )

  $length = $Text.Length
  if ($OpenParenIndex -lt 0 -or $OpenParenIndex -ge $length) {
    return $null
  }

  if ($Text[$OpenParenIndex] -ne '(') {
    return $null
  }

  $depth = 1
  $pos = $OpenParenIndex + 1

  while ($pos -lt $length -and $depth -gt 0) {
    $char = $Text[$pos]
    if ($char -eq '(') { $depth++ }
    elseif ($char -eq ')') { $depth-- }
    $pos++
  }

  if ($depth -ne 0) {
    return $null
  }

  $start = $OpenParenIndex + 1
  $innerLength = $pos - $start - 1
  if ($innerLength -lt 0) { $innerLength = 0 }
  $inner = if ($innerLength -gt 0) { $Text.Substring($start, $innerLength) } else { '' }
  $args = _Split-TopLevel -Text $inner -Separator ','
  for ($i = 0; $i -lt $args.Length; $i++) {
    $args[$i] = $args[$i].Trim()
  }

  [pscustomobject]@{
    Args = $args
    NextIndex = $pos
  }
}

function _Rewrite-ClipExpression {
  param(
    [Parameter(Mandatory = $true)][string]$Text
  )

  if ([string]::IsNullOrWhiteSpace($Text)) {
    return $Text
  }

  $builder = [System.Text.StringBuilder]::new()
  $length = $Text.Length
  $index = 0

  while ($index -lt $length) {
    $remaining = $length - $index
    if ($remaining -ge 4) {
      $segment = $Text.Substring($index, 4)
      if ($segment.Equals('clip', [System.StringComparison]::OrdinalIgnoreCase)) {
        $prevIndex = $index - 1
        $prevValid = ($prevIndex -lt 0) -or -not ([char]::IsLetterOrDigit($Text[$prevIndex]) -or $Text[$prevIndex] -eq '_')
        $probe = $index + 4
        while ($probe -lt $length -and [char]::IsWhiteSpace($Text[$probe])) { $probe++ }
        if ($prevValid -and $probe -lt $length -and $Text[$probe] -eq '(') {
          $parsed = _Parse-ClipInvocation -Text $Text -OpenParenIndex $probe
          if ($null -ne $parsed -and $parsed.Args.Length -eq 3) {
            $valueExpr = _Rewrite-ClipExpression $parsed.Args[0]
            $minExpr = _Rewrite-ClipExpression $parsed.Args[1]
            $maxExpr = _Rewrite-ClipExpression $parsed.Args[2]
            $replacement = "min(max(($valueExpr),($minExpr)),($maxExpr))"
            [void]$builder.Append($replacement)
            $index = $parsed.NextIndex
            continue
          }
        }
      }
    }

    [void]$builder.Append($Text[$index])
    $index++
  }

  return $builder.ToString()
}

function Expand-Clip {
  param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Expression,
    [string]$Min,
    [string]$Max
  )

  if ($PSBoundParameters.ContainsKey('Min') -or $PSBoundParameters.ContainsKey('Max')) {
    if (-not ($PSBoundParameters.ContainsKey('Min') -and $PSBoundParameters.ContainsKey('Max'))) {
      throw 'Expand-Clip requires both Min and Max when one is provided.'
    }

    $inner = if ([string]::IsNullOrWhiteSpace($Expression)) { '0' } else { $Expression }
    $minExpr = if ([string]::IsNullOrWhiteSpace($Min)) { '0' } else { $Min }
    $maxExpr = if ([string]::IsNullOrWhiteSpace($Max)) { '0' } else { $Max }
    $Expression = "clip(($inner),$minExpr,$maxExpr)"
  }

  if ([string]::IsNullOrWhiteSpace($Expression)) {
    return $Expression
  }

  return _Rewrite-ClipExpression -Text $Expression
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
        if ($depth -gt 0) {
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

Export-ModuleMember -Function \
  Convert-SciToDecimal, \
  Sanitize-Expr, \
  SubN, \
  Expand-Clip, \
  Escape-Commas-In-Parens
