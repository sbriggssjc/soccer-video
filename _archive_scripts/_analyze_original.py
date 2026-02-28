"""Analyze the original pipeline's crop path to understand what worked."""
import csv

diag = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00__portrait.diag.csv"

sources = {}
cx_values = []
with open(diag) as f:
    reader = csv.DictReader(f)
    for row in reader:
        src = row.get("source", "?")
        sources[src] = sources.get(src, 0) + 1
        fi = int(row["frame"])
        cx = float(row["cam_cx"])
        bx = float(row["ball_x"])
        cw = float(row["crop_w"])
        bic = row["ball_in_crop"]
        cx_values.append(cx)
        if fi % 50 == 0 or fi < 5:
            print(f"f{fi:3d}: cx={cx:7.1f} ball_x={bx:7.1f} crop_w={cw:.0f} src={src:10s} bic={bic}")

print(f"\nSource breakdown: {sources}")
print(f"Crop CX range: {min(cx_values):.0f} - {max(cx_values):.0f}")
print(f"Crop CX movement: {max(cx_values)-min(cx_values):.0f}px")
