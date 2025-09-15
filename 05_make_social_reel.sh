#!/bin/bash
# Build a vertical or square reel from top highlight clips.
TARGET_AR=${TARGET_AR:-9:16}
MAX_LEN=${MAX_LEN:-90}
BITRATE=${BITRATE:-6M}
CSV=out/highlights.csv
CLIPDIR=out/clips
OUT=out/social_reel.mp4
MUSIC=${MUSIC:-}
N=${1:-10}

# Optional constant A/V sync offset in seconds. Positive delays audio,
# negative makes it start earlier. Example: audioOffset=0.08
audioOffset=${audioOffset:-0}

set -e

[ -f "$CSV" ] || { echo "Missing $CSV"; exit 1; }
mkdir -p out
list=$(mktemp)
trap 'rm -f "$list"' EXIT

tail -n +2 "$CSV" | sort -t, -k3 -nr | head -n $N |
  nl -w4 -n rz | while IFS=$'\t,' read -r idx start end score; do
    printf "file '%s/clip_%04d.mp4'\n" "$CLIPDIR" "$idx" >> "$list"
  done

if [ "$TARGET_AR" = "9:16" ]; then
  VF="scale=-2:1920,crop=1080:1920,setsar=1,drawtext=font=Sans:text='Highlights':x=(w-text_w)/2:y=40:fontsize=64:fontcolor=white:enable='lt(t,2)'"
else
  VF="scale=1080:-2,pad=1080:1080:(1080-iw)/2:(1080-ih)/2,setsar=1,drawtext=font=Sans:text='Highlights':x=(w-text_w)/2:y=40:fontsize=64:fontcolor=white:enable='lt(t,2)'"
fi

# Build the reel into a temporary file first.
TMP="${OUT}.tmp"
if [ -n "$MUSIC" ]; then
  ffmpeg -y -f concat -safe 0 -i "$list" -i "$MUSIC" \
    -filter_complex "[0:v]$VF[v];[0:a][1:a]amix=inputs=2:duration=shortest[a]" \
    -map '[v]' -map '[a]' -c:v libx264 -b:v $BITRATE -c:a aac -t $MAX_LEN "$TMP"
else
  ffmpeg -y -f concat -safe 0 -i "$list" \
    -vf "$VF" -c:v libx264 -b:v $BITRATE -c:a aac -t $MAX_LEN "$TMP"
fi

# Apply audio offset and finalize output.
ffmpeg -y -i "$TMP" -itsoffset "$audioOffset" -i "$TMP" -map 0:v -map 1:a -c copy "$OUT"
rm -f "$TMP"
