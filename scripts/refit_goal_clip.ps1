$cfg = "C:\Users\scott\soccer-video\configs\zoom.yaml"
$csv = "C:\Users\scott\soccer-video\out\autoframe_work\004__GOAL__t266.50-t283.10_zoom.csv"
$out = "C:\Users\scott\soccer-video\out\autoframe_work\004__GOAL__t266.50-t283.10_zoom.ps1vars"
Remove-Item $out -Force -ErrorAction SilentlyContinue

python .\fit_expr.py `
  --csv "$csv" `
  --out "$out" `
  --degree 3 `
  --profile landscape `
  --roi goal `
  --config "$cfg" `
  --lead-ms 16 `                 # smaller lead so we don’t pan past receivers
  --alpha-slow 0.06 `
  --alpha-fast 0.70 `            # gentler fast-smoothing (less whipsaw)
  --deadzone 10 `                # a little more deadzone to suppress micro jitter
  --boot-wide-ms 4200 `          # stay wide until the play “settles”
  --snap-accel-th 0.16 `         # snap on milder accelerations
  --snap-widen 0.50 `            # when snapping, also widen a lot
  --snap-decay-ms 380 `
  --snap-hold-ms 220 `
  --z-tight 1.65 `               # MUCH safer tight bound (no super-tight crop)
  --z-wide 1.04 `
  --zoom-tighten-rate 0.010 `
  --zoom-widen-rate 0.160 `      # widen fast whenever confidence/motion says so
  --v-enabled `
  --v-gain 0.85 `
  --v-deadzone 10 `
  --v-top-margin 60 `
  --v-bottom-margin 60 `
  --goal-bias 0.60 `             # pull toward goal mouth (it’s a GOAL clip)
  --conf-th 0.70 `               # ignore low-confidence junk in the CSV
  --celebration-ms 2200 `
  --celebration-tight 2.20       # celebration still a bit tight but not extreme
