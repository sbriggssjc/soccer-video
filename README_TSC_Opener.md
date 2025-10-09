# TSC Portrait Opener (Locked-In Version)

**Source of truth.** This README + scripts generate the portrait opener that matches the team’s tail background and uses two official badges at **identical size/position**. The only thing that changes is the player’s face inside the circular hole.

## Assets (must exist)
- Background (same as tail):  
  `C:\Users\scott\soccer-video\brand\tsc\end_card_1080x1920.png`
- Solid badge (big logo):  
  `C:\Users\scott\soccer-video\brand\tsc\badge_clean.png`
- Hole badge (transparent middle):  
  `C:\Users\scott\soccer-video\brand\tsc\badge_hole.png`

## Why this is locked
- **Identical logo size**: We scale the solid badge to `BadgeW` and then force the hole art to match **exactly** with `scale2ref`.
- **No small tail logo during opener**: The big badge fades from SOLID → HOLE once; the HOLE remains to the end.
- **Face only changes**: The player photo is scaled/cropped to the inner circle, centered for head & shoulders.

## Defaults
- Duration: 6.0s; crossfade at t=1.0s for 1.0s
- Geometry: `BadgeW=900`, `BadgeY=520`, `HoleScale=0.58` (inner circle), `FaceYOffset=-30`
- Font: `C:/WINDOWS/Fonts/arialbd.ttf`

## Single render
```powershell
pwsh -File scripts\make_opener.ps1 `
  -PlayerName "First Last" `
  -PlayerNumber 99 `
  -PlayerPhoto "C:\path\to\photo.jpg" `
  -OutDir "C:\Users\scott\soccer-video\out\opener"
```
Batch render
Copy roster_template.csv to your own roster.csv, fill name/number/photo.

Run:

powershell
Copy code
pwsh -File scripts\batch_make_openers.ps1 `
  -RosterCsv "C:\path\to\roster.csv" `
  -OutDir "C:\Users\scott\soccer-video\out\opener"
Cleanup outputs
Preview first, then run for real:

powershell
Copy code
pwsh -File scripts\cleanup_repo.ps1 -TargetDir "C:\Users\scott\soccer-video\out\opener" -WhatIf
pwsh -File scripts\cleanup_repo.ps1 -TargetDir "C:\Users\scott\soccer-video\out\opener"
Notes
If the hole art is microscopically off-center, adjust face only with -HoleDX/-HoleDY.

Keep brand assets in place; scripts assume those absolute paths.
