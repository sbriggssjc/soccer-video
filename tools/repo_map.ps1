<# 
  repo_map.ps1 — inventory your toolkit and capture CLI help
  Output:
    - out\repo_map.md  (pretty markdown index)
    - out\repo_map_commands.csv (flat list of discovered commands)
#>
param(
  [string]$Root = ".",
  [string]$OutDir = ".\out"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$md = Join-Path $OutDir "repo_map.md"
$csv = Join-Path $OutDir "repo_map_commands.csv"

# --- basic repo stats ---
$files = Get-ChildItem $Root -Recurse -File -ErrorAction SilentlyContinue
$sizeGB = [math]::Round(($files | Measure-Object Length -Sum).Sum / 1GB, 3)

# --- find interesting files ---
$py = $files | Where-Object {$_.Extension -in ".py"} | Sort-Object FullName
$ps = $files | Where-Object {$_.Extension -in ".ps1",".psm1"} | Sort-Object FullName
$yaml = $files | Where-Object {$_.Extension -in ".yml",".yaml"}
$readmes = $files | Where-Object { $_.Name -match '^README' }

# --- helper: safe run & capture (for CLI help, etc.) ---
function Try-Run([string]$CmdLine, [int]$TimeoutSec=25) {
  try {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "powershell"
    $psi.Arguments = "-NoProfile -Command $CmdLine"
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $p = [System.Diagnostics.Process]::Start($psi)
    if (-not $p.WaitForExit($TimeoutSec*1000)) { $p.Kill() | Out-Null }
    $out = $p.StandardOutput.ReadToEnd()
    if (-not $out) { $out = $p.StandardError.ReadToEnd() }
    return $out
  } catch {
    return "ERROR running: $CmdLine`n$($_.Exception.Message)"
  }
}

# --- discover top-level CLI(s) in this project ---
# Heuristics: pyproject console_scripts, soccerhl package, and direct scripts with argparse/click/typer
$consoleScripts = @()
$pyproject = Get-ChildItem $Root -Recurse -Filter "pyproject.toml" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($pyproject) {
  $txt = Get-Content -Raw $pyproject.FullName
  if ($txt -match '(?ms)\\[project\\.scripts\\](.*?)\\n\\[') {
    $block = $Matches[1]
    $consoleScripts += ($block -split "`n" | ForEach-Object {
      ($_ -replace '#.*','').Trim()
    } | Where-Object { $_ -match '^\s*([\w\-]+)\s*=' } | ForEach-Object {
      ($_ -replace '=.*','').Trim()
    })
  }
}

# Always try 'soccerhl' (your main CLI)
if ($consoleScripts -notcontains "soccerhl") { $consoleScripts += "soccerhl" }

$consoleScripts = $consoleScripts | Sort-Object -Unique

# --- parse subcommands from --help output ---
$cmdRows = New-Object System.Collections.Generic.List[object]
$helpMap = @{}
foreach ($cmd in $consoleScripts) {
  $h = Try-Run "$cmd --help"
  $helpMap[$cmd] = $h
  # extract lines under a 'Commands:' or similar section
  $subs = @()
  foreach ($line in ($h -split "`n")) {
    if ($line -match '^\s{0,4}Commands?:\s*$') { $seen = $true; continue }
    if ($seen) {
      if ($line -match '^\s{0,4}\w') {
        # Expect "name  description"
        if ($line -match '^\s*([A-Za-z0-9\-_]+)\s{2,}(.+)$') {
          $subs += [pscustomobject]@{ sub=$Matches[1]; desc=$Matches[2] }
        } elseif ($line -match '^\s*([A-Za-z0-9\-_]+)\s*$') {
          $subs += [pscustomobject]@{ sub=$Matches[1]; desc="" }
        }
      } elseif ($line -match '^\s*$') {
        # blank line might end block; keep scanning a bit
      }
    }
  }
  if ($subs.Count -eq 0) {
    # fallback: known subcmds often used
    $subs = @(
      [pscustomobject]@{sub='detect'; desc='detect highlight windows'},
      [pscustomobject]@{sub='shrink'; desc='tighten windows around peaks'},
      [pscustomobject]@{sub='clips';  desc='export per-window MP4s'},
      [pscustomobject]@{sub='topk';   desc='rank & pick best plays'},
      [pscustomobject]@{sub='reel';   desc='render final reel'}
    )
  }

  foreach ($s in $subs) {
    # capture sub-help
    $subHelp = Try-Run "$cmd $($s.sub) --help"
    $helpMap["$cmd $($s.sub)"] = $subHelp
    $cmdRows.Add([pscustomobject]@{
      command = $cmd
      subcommand = $s.sub
      description = $s.desc
    })
  }
}

# --- scan python for argparse/click/typer entry points ---
$entryRows = New-Object System.Collections.Generic.List[object]
foreach ($p in $py) {
  $head = (Get-Content $p.FullName -TotalCount 400) -join "`n"
  $hasMain  = $head -match '__main__'
  $hasArgp  = $head -match 'argparse'
  $hasClick = $head -match '\bclick\b'
  $hasTyper = $head -match '\btyper\b'
  if ($hasMain -or $hasArgp -or $hasClick -or $hasTyper) {
    $entryRows.Add([pscustomobject]@{
      path = $p.FullName
      argparse = $hasArgp
      click   = $hasClick
      typer   = $hasTyper
      main    = $hasMain
    })
  }
}

# --- write CSV of commands ---
$cmdRows | Export-Csv -Path $csv -NoTypeInformation -Encoding UTF8

# --- build markdown ---
$sb = New-Object System.Text.StringBuilder
$null = $sb.AppendLine("# Repository Map")
$null = $sb.AppendLine()
$null = $sb.AppendLine("*Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')*")
$null = $sb.AppendLine()
$null = $sb.AppendLine("**Files:** $($files.Count)  •  **Size:** ${sizeGB} GB")
$null = $sb.AppendLine()
$null = $sb.AppendLine("## Key Files")
$null = $sb.AppendLine()
if ($readmes) {
  $null = $sb.AppendLine("**READMEs**")
  foreach ($r in $readmes) { $null = $sb.AppendLine("- `$(Resolve-Path $r.FullName -Relative)`") }
  $null = $sb.AppendLine()
}
if ($yaml) {
  $null = $sb.AppendLine("**YAML / config**")
  foreach ($y in ($yaml | Sort-Object FullName)) { $null = $sb.AppendLine("- `$(Resolve-Path $y.FullName -Relative)`") }
  $null = $sb.AppendLine()
}
$null = $sb.AppendLine("**PowerShell scripts**")
foreach ($f in $ps) { $null = $sb.AppendLine("- `$(Resolve-Path $f.FullName -Relative)`") }
$null = $sb.AppendLine()
$null = $sb.AppendLine("**Python files with CLI hints (argparse/click/typer or __main__)**")
foreach ($e in $entryRows) { $null = $sb.AppendLine("- `$(Resolve-Path $e.path -Relative)`  (argparse=$($e.argparse), click=$($e.click), typer=$($e.typer), main=$($e.main))") }
$null = $sb.AppendLine()

$null = $sb.AppendLine("## Discovered Console Scripts")
foreach ($cmd in $consoleScripts) {
  $null = $sb.AppendLine()
  $null = $sb.AppendLine($"### { $cmd }")
  $null = $sb.AppendLine()
  $h = $helpMap[$cmd] -replace '``','`'   # tame backticks
  $null = $sb.AppendLine("````")
  $null = $sb.AppendLine($h.Trim())
  $null = $sb.AppendLine("````")

  # subcommands
  $subs = $cmdRows | Where-Object { $_.command -eq $cmd }
  if ($subs) {
    $null = $sb.AppendLine()
    $null = $sb.AppendLine("#### Subcommands")
    foreach ($s in $subs) {
      $null = $sb.AppendLine()
      $null = $sb.AppendLine($"**{ $s.subcommand }** — { $s.description }")
      $sh = $helpMap["$cmd $($s.subcommand)"] -replace '``','`'
      $null = $sb.AppendLine()
      $null = $sb.AppendLine("````")
      $null = $sb.AppendLine($sh.Trim())
      $null = $sb.AppendLine("````")
    }
  }
}

# recent files
$null = $sb.AppendLine()
$null = $sb.AppendLine("## Recently Modified (last 7 days)")
$recent = $files | Where-Object { $_.LastWriteTime -gt (Get-Date).AddDays(-7) } | Sort-Object LastWriteTime -Descending | Select-Object -First 40
foreach ($f in $recent) {
  $rel = (Resolve-Path $f.FullName -Relative)
  $null = $sb.AppendLine("- $($f.LastWriteTime.ToString('yyyy-MM-dd HH:mm'))  `$rel`  ($([math]::Round($f.Length/1KB,1)) KB)")
}

# write it
[IO.File]::WriteAllText($md, $sb.ToString(), [Text.Encoding]::UTF8)

"✓ Wrote: $md"
"✓ Wrote: $csv"
