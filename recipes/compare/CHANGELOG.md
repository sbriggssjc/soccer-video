# Compare Recipe Changelog

This recipe is intentionally tiny and "frozen" so that we can sanity-check the
Autoframe math at any time on any workstation.

## 2025-09-24

* Use `ih` (input height) in every expression so the crop scales correctly
  regardless of the clip dimensions.
* Sanitize FFmpeg expressions by stripping whitespace and converting scientific
  notation (for example `1.2e-5` → `(1.2*pow(10,-5))`).
* Replace any use of `n` with `t*FPS` before writing the VF so that the preview
  behaves identically to ffmpeg's runtime evaluation.
* Emit UTF-8 **without BOM** for the temporary `.vf` file to keep ffmpeg happy.
* Require `ffmpeg`/`ffprobe` up front and bubble their failure codes so the
  script exits non-zero if rendering fails.
* Keep a deterministic sample clip (`samples/clip.mp4`) alongside a matching
  set of expression vars so the harness command:

  ```powershell
  .\recipes\compare\Run-Compare.ps1 `
    -In .\recipes\compare\samples\clip.mp4 `
    -Vars .\recipes\compare\samples\clip_zoom.ps1vars `
    -Out .\recipes\compare\samples\out.mp4 `
    -VF .\recipes\compare\samples\tmp.vf
  ```

  can be used as a quick "does it still work?" smoke test.
