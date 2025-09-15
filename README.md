# soccer-video

Scripts for building soccer highlight clips and social reels.

## Quick A/V sync knob

If the commentary is slightly out of sync, adjust the `audioOffset` environment
variable (in seconds) when running `05_make_social_reel.sh` and re-run.

- **Audio late** (you see the kick before you hear it): `audioOffset=-0.08`
- **Audio early**: `audioOffset=0.08`

Values in the ±0.06–0.12 range usually work best.
