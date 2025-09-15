# soccer-video

Tools for detecting exciting moments and assembling highlight reels from a full game recording.

If a final `ffmpeg` concat ever prints a `Non-monotonous DTS` warning, re-encode the list instead of stream-copying:

```
ffmpeg -hide_banner -loglevel warning -y -safe 0 -f concat -i list.txt \
  -c:v libx264 -preset veryfast -crf 20 -c:a aac -b:a 160k \
  out/smart10_clean_zoom.mp4
```

`list.txt` should contain `file` lines pointing at the parts to concatenate.

