# Examples

The `generate_sample.py` script creates a short synthetic match clip that is small
enough for regression tests. Run it from PowerShell with:

```powershell
python .\examples\generate_sample.py
```

After generating the clip you can process it end-to-end using:

```powershell
soccerhl detect --config .\examples\sample_config.yaml
soccerhl shrink --config .\examples\sample_config.yaml --csv .\examples\out\highlights.csv --out .\examples\out\highlights_smart.csv
soccerhl clips --config .\examples\sample_config.yaml --csv .\examples\out\highlights_smart.csv --outdir .\examples\out\clips
soccerhl topk --config .\examples\sample_config.yaml --candirs .\examples\out\clips
soccerhl reel --config .\examples\sample_config.yaml --list .\examples\out\smart_top10_concat.txt --out .\examples\out\reels\sample_top10.mp4
```

All commands work on Windows PowerShell without line continuations.
