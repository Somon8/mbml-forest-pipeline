"""spatial_features.py — Append neighbourhood / directional spatial features
to the preprocessed CSV.

Input (default):  out_10km_idx_preprocessed.csv
Output (default): same path (overwrite). Use --out <path> to redirect.

The preprocessed CSV is a complete 801 × 801 regular grid on EPSG:3006 at
12.5 m spacing, with cell-center coordinates in `x, y`. We pivot the target
column and the two exclusion masks (`is_no_forest`, `is_lake`) to 2D arrays
— row 0 = northernmost (largest y), column 0 = westernmost (smallest x) —
run convolutions and distance transforms on the arrays, then flatten back
onto the dataframe by matching `(x, y)`.

The same eight per-layer features are computed for each layer in
`TARGET_LAYERS` (currently `p95_omdrev2` and `medelhojd_omdrev2` — either
is a plausible choice for the modelling target y, so we build both now).
The two distance-to-mask features are target-agnostic and computed once.

Columns added, per target layer `<L>`:

    <L>_mean3, <L>_std3            3×3 neighbourhood mean and std  (dm)
    <L>_mean5, <L>_std5            5×5 neighbourhood mean and std  (dm)
    <L>_dNS                        mean(south 3 cells) − mean(north 3 cells) (dm)
                                   positive → taller canopy south of the pixel
                                   (so the pixel itself is shaded from the south)
    <L>_dEW                        mean(east) − mean(west)                (dm)
    <L>_grad_mag                   |∇L|, Sobel                            (dm / cell)
    <L>_grad_aspect                atan2(dZ/dy_geo, dZ/dx_geo)            (radians,
                                   0 = east, π/2 = north)

Plus once, globally:

    dist_to_no_forest_m            Euclidean distance to nearest `is_no_forest`   (m)
    dist_to_lake_m                 Euclidean distance to nearest `is_lake` pixel  (m)

Boundary handling: convolutions use `reflect`. Distance transforms treat
cells outside the AOI as unknown (not a neighbour).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage


CELL_M = 12.5
TARGET_LAYERS = ["p95_omdrev2", "medelhojd_omdrev2"]


def per_layer_cols(layer):
    return [
        f"{layer}_mean3", f"{layer}_std3",
        f"{layer}_mean5", f"{layer}_std5",
        f"{layer}_dNS", f"{layer}_dEW",
        f"{layer}_grad_mag", f"{layer}_grad_aspect",
    ]


NEW_COLS = [c for L in TARGET_LAYERS for c in per_layer_cols(L)] + [
    "dist_to_no_forest_m", "dist_to_lake_m",
]


def pivot_to_grid(df, col):
    """Pivot `col` to a 2D array with row 0 = north, column 0 = west."""
    piv = (
        df.pivot(index="y", columns="x", values=col)
        .sort_index(ascending=False, axis=0)
        .sort_index(ascending=True, axis=1)
    )
    return piv.values


def window_mean_std(arr, size):
    mean = ndimage.uniform_filter(arr, size=size, mode="reflect")
    mean_sq = ndimage.uniform_filter(arr * arr, size=size, mode="reflect")
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    return mean, np.sqrt(var)


def directional_diff(arr):
    """(dNS, dEW) = (south_mean − north_mean, east_mean − west_mean)."""
    k_n = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=float) / 3.0
    k_s = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=float) / 3.0
    k_e = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=float) / 3.0
    k_w = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float) / 3.0
    n = ndimage.convolve(arr, k_n, mode="reflect")
    s = ndimage.convolve(arr, k_s, mode="reflect")
    e = ndimage.convolve(arr, k_e, mode="reflect")
    w = ndimage.convolve(arr, k_w, mode="reflect")
    return s - n, e - w


def gradient_mag_aspect(arr):
    # row 0 is north → dZ/dy_geo has opposite sign of the row-axis Sobel.
    dz_dr = ndimage.sobel(arr, axis=0, mode="reflect")
    dz_dc = ndimage.sobel(arr, axis=1, mode="reflect")
    dz_dx = dz_dc
    dz_dy = -dz_dr
    return np.hypot(dz_dx, dz_dy), np.arctan2(dz_dy, dz_dx)


def distance_to_mask(mask, cell_m=CELL_M):
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return ndimage.distance_transform_edt(~mask) * cell_m


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv_in", nargs="?", default="out_10km_idx_preprocessed.csv")
    ap.add_argument("--out", default=None,
                    help="Output path (default: overwrite input).")
    args = ap.parse_args()

    csv_in = Path(args.csv_in)
    csv_out = Path(args.out) if args.out else csv_in

    print(f"Reading {csv_in} ...")
    df = pd.read_csv(csv_in)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Idempotent: drop any spatial columns from a previous run before recomputing.
    existing = [c for c in NEW_COLS if c in df.columns]
    if existing:
        print(f"  dropping existing spatial columns: {existing}")
        df = df.drop(columns=existing)

    nx, ny = df.x.nunique(), df.y.nunique()
    assert nx * ny == len(df), f"grid not complete: {nx} × {ny} ≠ {len(df):,}"
    print(f"  grid: {ny} rows × {nx} cols")

    # 2D arrays -----------------------------------------------------------------
    mask_nf = pivot_to_grid(df, "is_no_forest").astype(bool)
    mask_lk = pivot_to_grid(df, "is_lake").astype(bool)

    grids = {}
    for L in TARGET_LAYERS:
        layer = pivot_to_grid(df, L).astype(np.float64)
        m3, s3 = window_mean_std(layer, 3)
        m5, s5 = window_mean_std(layer, 5)
        dNS, dEW = directional_diff(layer)
        gmag, gasp = gradient_mag_aspect(layer)
        grids.update({
            f"{L}_mean3":        m3,
            f"{L}_std3":         s3,
            f"{L}_mean5":        m5,
            f"{L}_std5":         s5,
            f"{L}_dNS":          dNS,
            f"{L}_dEW":          dEW,
            f"{L}_grad_mag":     gmag,
            f"{L}_grad_aspect":  gasp,
        })

    grids["dist_to_no_forest_m"] = distance_to_mask(mask_nf)
    grids["dist_to_lake_m"]      = distance_to_mask(mask_lk)

    # Flatten 2D grids back onto df rows -----------------------------------------
    ys = np.sort(df.y.unique())[::-1]   # row 0 = largest y  (north)
    xs = np.sort(df.x.unique())         # col 0 = smallest x (west)
    row_of_y = pd.Series(np.arange(len(ys)), index=ys)
    col_of_x = pd.Series(np.arange(len(xs)), index=xs)
    ri = row_of_y.reindex(df.y).to_numpy()
    ci = col_of_x.reindex(df.x).to_numpy()

    for name, grid in grids.items():
        df[name] = grid[ri, ci]

    print(f"Writing {csv_out} ...")
    df.to_csv(csv_out, index=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")
    print("New columns (p50 / p95 / max):")
    for name in grids:
        v = df[name].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {name:34s}  {v.median():8.3f}   {v.quantile(0.95):8.3f}   {v.max():8.3f}")


if __name__ == "__main__":
    main()
