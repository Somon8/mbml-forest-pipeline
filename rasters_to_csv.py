"""
rasters_to_csv.py — Combine a folder of GeoTIFFs into a flat CSV.

Usage:
    python rasters_to_csv.py <input_folder> <output.csv> [options]

Options:
    --crop minx miny maxx maxy   Bounding box in raster CRS (e.g. EPSG:3006)
    --sw LAT LON                 SW corner in WGS84 (as given by Google Maps)
    --size METERS                Side length of the square extending N and E from --sw (default 10000)
    --aggregate N                Average/mode over N×N pixel blocks
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds, Window


def parse_args():
    p = argparse.ArgumentParser(description="Merge GeoTIFFs into a CSV.")
    p.add_argument("input_folder", help="Folder containing .tif files")
    p.add_argument("output_csv", help="Output CSV path")
    p.add_argument("--crop", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"),
                   help="Crop to bounding box in raster CRS")
    p.add_argument("--sw", nargs=2, type=float, metavar=("LAT", "LON"),
                   help="SW corner in WGS84 lat/lon (e.g. from Google Maps)")
    p.add_argument("--size", type=float, default=10000.0, metavar="METERS",
                   help="Side length of the square extending N and E from --sw (default 10000)")
    p.add_argument("--aggregate", type=int, metavar="N",
                   help="Aggregate N×N pixel blocks (mean for float, mode for int)")
    args = p.parse_args()
    if args.sw and args.crop:
        p.error("use either --crop or --sw, not both")
    return args


def bbox_from_sw(lat, lon, size, ref_crs, ref_transform):
    """SW corner (WGS84) + size(m) → bbox in ref_crs, expanded to enclose any touched pixel."""
    from pyproj import Transformer
    t = Transformer.from_crs(4326, ref_crs, always_xy=True)
    sw_x, sw_y = t.transform(lon, lat)
    minx, miny = sw_x, sw_y
    maxx, maxy = sw_x + size, sw_y + size
    # Snap outward to the raster grid so any pixel the square overlaps is included.
    px, py = abs(ref_transform.a), abs(ref_transform.e)
    ox, oy = ref_transform.c, ref_transform.f
    import math
    minx = ox + math.floor((minx - ox) / px) * px
    maxx = ox + math.ceil((maxx - ox) / px) * px
    maxy = oy - math.floor((oy - maxy) / py) * py
    miny = oy - math.ceil((oy - miny) / py) * py
    return minx, miny, maxx, maxy


def read_band(path, window=None):
    """Open a raster and read band 1 (with optional window). Returns (data, profile)."""
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        profile["count"] = 1
        if window is not None:
            data = src.read(1, window=window)
            profile["transform"] = src.window_transform(window)
            profile["height"] = data.shape[0]
            profile["width"] = data.shape[1]
        else:
            data = src.read(1)
        return data.astype(np.float64), profile


def reproject_to_ref(data, src_profile, ref_profile, is_categorical):
    """Reproject data to match ref_profile grid."""
    resampling = Resampling.nearest if is_categorical else Resampling.bilinear
    dst = np.empty((ref_profile["height"], ref_profile["width"]), dtype=np.float64)
    dst[:] = np.nan

    reproject(
        source=data,
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        resampling=resampling,
        src_nodata=src_profile.get("nodata"),
        dst_nodata=np.nan,
    )
    return dst


def aggregate_block(data, N, is_categorical, nodata_mask):
    """Reduce data by N×N blocks. nodata_mask=True where pixel is nodata."""
    h, w = data.shape
    h2, w2 = h // N, w // N
    data_crop = data[:h2 * N, :w2 * N].copy()
    mask_crop = nodata_mask[:h2 * N, :w2 * N]

    data_crop[mask_crop] = np.nan
    blocks = data_crop.reshape(h2, N, w2, N)

    if is_categorical:
        from scipy import stats
        out = np.empty((h2, w2), dtype=np.float64)
        out[:] = np.nan
        for i in range(h2):
            for j in range(w2):
                block = blocks[i, :, j, :].ravel()
                valid = block[~np.isnan(block)]
                if len(valid) > 0:
                    out[i, j] = stats.mode(valid, keepdims=False).mode
        return out
    else:
        with np.errstate(all="ignore"):
            return np.nanmean(blocks, axis=(1, 3))


def cell_centers(transform, height, width):
    """Return (x_coords, y_coords) arrays of cell centers."""
    cols = np.arange(width)
    rows = np.arange(height)
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    return xs, ys


def main():
    args = parse_args()

    folder = Path(args.input_folder)
    if not folder.is_dir():
        sys.exit(f"Error: '{folder}' is not a directory.")

    tif_files = sorted(folder.glob("*.tif"))
    if not tif_files:
        sys.exit(f"Error: No .tif files found in '{folder}'.")

    print(f"Found {len(tif_files)} raster(s):\n")

    # ── Inspect all rasters ────────────────────────────────────────────────
    raster_infos = []
    for path in tif_files:
        with rasterio.open(path) as src:
            info = {
                "path": path,
                "name": path.stem,
                "crs": src.crs,
                "res": src.res,
                "bounds": src.bounds,
                "nodata": src.nodata,
                "dtype": src.dtypes[0],
                "transform": src.transform,
                "height": src.height,
                "width": src.width,
            }
            raster_infos.append(info)
            print(f"  {path.name}")
            print(f"    CRS      : {src.crs}")
            print(f"    Resolution: {src.res}")
            print(f"    Extent   : {src.bounds}")
            print(f"    Nodata   : {src.nodata}")
            print(f"    Dtype    : {src.dtypes[0]}")
            print()

    ref = raster_infos[0]

    # ── Resolve --sw/--size into a --crop bbox ────────────────────────────
    if args.sw:
        lat, lon = args.sw
        args.crop = list(bbox_from_sw(lat, lon, args.size, ref["crs"], ref["transform"]))
        print(f"SW {lat},{lon} + {args.size} m square → crop {args.crop}\n")

    # ── Crop window for reference raster ──────────────────────────────────
    crop_window = None
    if args.crop:
        minx, miny, maxx, maxy = args.crop
        with rasterio.open(ref["path"]) as src:
            crop_window = window_from_bounds(minx, miny, maxx, maxy, src.transform)
            crop_window = crop_window.round_offsets().round_lengths()
        print(f"Cropping to bbox: {args.crop}\n")

    # ── Load reference raster ─────────────────────────────────────────────
    ref_data, ref_profile = read_band(ref["path"], window=crop_window)
    ref_nodata = ref["nodata"]
    ref_is_cat = np.issubdtype(np.dtype(ref["dtype"]), np.integer)

    # Replace nodata with nan
    nodata_masks = {}
    if ref_nodata is not None:
        nodata_masks[ref["name"]] = ref_data == ref_nodata
        ref_data[nodata_masks[ref["name"]]] = np.nan
    else:
        nodata_masks[ref["name"]] = np.zeros(ref_data.shape, dtype=bool)

    layers = {ref["name"]: (ref_data, ref_is_cat)}
    reprojected = []

    # ── Load (and optionally reproject) other rasters ─────────────────────
    for info in raster_infos[1:]:
        is_cat = np.issubdtype(np.dtype(info["dtype"]), np.integer)
        needs_reproj = (
            info["crs"] != ref["crs"]
            or info["res"] != ref["res"]
            or info["bounds"] != ref["bounds"]
        )

        if needs_reproj:
            # Read only the source window that covers the crop bbox (avoids loading full raster)
            with rasterio.open(info["path"]) as src:
                if args.crop:
                    # Transform crop bbox from ref CRS into source CRS
                    src_minx, src_miny, src_maxx, src_maxy = transform_bounds(
                        ref_profile["crs"], src.crs,
                        *[args.crop[i] for i in [0, 1, 2, 3]]
                    )
                    src_window = window_from_bounds(src_minx, src_miny, src_maxx, src_maxy, src.transform)
                    src_window = src_window.round_offsets().round_lengths()
                    # Add a small buffer (10 px) to avoid edge artefacts after reproject
                    row_off = max(0, int(src_window.row_off) - 10)
                    col_off = max(0, int(src_window.col_off) - 10)
                    row_end = min(src.height, int(src_window.row_off + src_window.height) + 10)
                    col_end = min(src.width,  int(src_window.col_off + src_window.width)  + 10)
                    src_window = Window(col_off, row_off, col_end - col_off, row_end - row_off)
                    data = src.read(1, window=src_window).astype(np.float64)
                    profile = src.profile.copy()
                    profile["transform"] = src.window_transform(src_window)
                    profile["height"] = data.shape[0]
                    profile["width"] = data.shape[1]
                else:
                    data = src.read(1).astype(np.float64)
                    profile = src.profile.copy()

            nd = info["nodata"]
            mask = (data == nd) if nd is not None else np.zeros(data.shape, dtype=bool)
            data[mask] = np.nan
            print(f"Reprojecting {info['name']} to match {ref['name']} ...")
            data = reproject_to_ref(data, profile, ref_profile, is_cat)
            mask = np.isnan(data)
            reprojected.append(info["name"])
        else:
            data, _ = read_band(info["path"], window=crop_window)
            nd = info["nodata"]
            mask = (data == nd) if nd is not None else np.zeros(data.shape, dtype=bool)
            data[mask] = np.nan

        nodata_masks[info["name"]] = mask
        layers[info["name"]] = (data, is_cat)

    # ── Aggregation ───────────────────────────────────────────────────────
    N = args.aggregate
    agg_transform = ref_profile["transform"]

    if N and N > 1:
        print(f"Aggregating {N}×{N} pixel blocks ...\n")
        # Build combined nodata mask before aggregating
        combined_nodata = np.zeros(ref_data.shape, dtype=bool)
        for name, mask in nodata_masks.items():
            combined_nodata |= mask

        aggregated = {}
        for name, (data, is_cat) in layers.items():
            aggregated[name] = aggregate_block(data, N, is_cat, nodata_masks[name])

        layers = {name: (arr, layers[name][1]) for name, arr in aggregated.items()}

        # Adjust transform for aggregated grid
        t = ref_profile["transform"]
        agg_transform = rasterio.transform.Affine(
            t.a * N, t.b, t.c,
            t.d, t.e * N, t.f
        )
        h2 = ref_profile["height"] // N
        w2 = ref_profile["width"] // N
        shape = (h2, w2)
    else:
        shape = (ref_profile["height"], ref_profile["width"])

    # ── Build coordinate arrays ───────────────────────────────────────────
    xs, ys = cell_centers(agg_transform, shape[0], shape[1])
    xx, yy = np.meshgrid(xs, ys)

    # ── Assemble DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame({"x": xx.ravel(), "y": yy.ravel()})
    for name, (data, _) in layers.items():
        arr = data[:shape[0], :shape[1]]
        df[name] = arr.ravel()

    # Drop rows where any layer is nodata (nan)
    layer_cols = list(layers.keys())
    before = len(df)
    df.dropna(subset=layer_cols, inplace=True)
    dropped = before - len(df)

    # ── Write CSV ─────────────────────────────────────────────────────────
    df.to_csv(args.output_csv, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    print("─" * 50)
    print(f"Output rows   : {len(df):,}  ({dropped:,} dropped as nodata)")
    print(f"CRS           : {ref['crs']}")
    print(f"Columns       : x, y, {', '.join(layer_cols)}")
    if reprojected:
        print(f"Reprojected   : {', '.join(reprojected)}")
    else:
        print("Reprojected   : none (all on same grid)")
    print(f"Written to    : {args.output_csv}")


if __name__ == "__main__":
    main()
