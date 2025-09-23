# Autoframe Polynomial Zooms

This helper analyzes a clip with OpenCV, fits smooth polynomial expressions for
pans/zoom, and pipes them into the existing FFmpeg crop chain.

## Usage

1. Generate coefficients with Farnebäck motion tracking:

   ```bash
   python scripts/autoframe.py in.mp4 zoom_coeffs.csv tiktok
   ```

   * `in.mp4` – source clip to analyze.
   * `zoom_coeffs.csv` – CSV storing the fitted polynomials (append-only).
   * `tiktok` – profile label (optional, defaults to `tiktok`).

2. Apply the coefficients during render:

   ```powershell
   pwsh pipeline/apply_zooms.ps1 -Input in.mp4 -Profile tiktok -Coeffs zoom_coeffs.csv
   ```

   The script falls back to the baked-in expressions when no matching row is
   found for the requested clip/profile.

The CSV stores columns `clip`, `profile`, `cx_poly`, `cy_poly`, and `z_poly`,
where each polynomial uses `n` as the frame index and `n*n` for the quadratic
term so that the strings can be dropped directly into FFmpeg expressions.
