# Data pipeline — Swedish forest rasters → tabular dataset

This document describes, end to end, how the tabular dataset used for modelling
is produced from the raw Skogsstyrelsen GeoTIFFs and the Sverige-indexruta
vector grid. It is intended to be citable in the project report.

## 1. Input data

All rasters in `combined_rasters/` are symlinked from source folders on
`/Volumes/WD2TB/MBML project data/`. They come from two providers:
**Skogsstyrelsen** (the Swedish Forest Agency) for the "Skogliga grunddata"
layers, soil moisture and flow accumulation, and **SLU** (Swedish University
of Agricultural Sciences) for the SLU Skogskarta species-specific layers.

Each file is a single-band GeoTIFF covering southern Sweden (Blekinge/Skåne
tile, index 12, except the 1 m `tradhojd` and `flodesackumulering` which are
delivered as larger / tiled files).

The tables below list **every layer in the current CSV**, its physical
meaning, native resolution, dtype, nodata sentinel, and summary statistics
computed over the 10 km AOI after reprojection to the reference 12.5 m grid
(641 601 cells).

**Unit conventions.** Heights (`medelhojd_*`, `p95_*`, `tradhojd`) are in
**decimeters** — divide by 10 to get metres. Diameter is in **cm**. Biomass
and volume are **per-hectare** (t/ha and m³/ha respectively). Basal area
(`grundyta_*`) is **m²/ha**. Vegetation ratio is a **%** (0–100).

### Skogliga grunddata — cycle 2 (omdrev 2, 2018–2025, 10 m)

Skogsstyrelsen's airborne-laser-derived forest metrics, second inventory
cycle.

| File | Variable | Units | Res. | Dtype | Nodata | zeros% | p50 | p95 | max |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `biomassa_omdrev2` | Above-ground biomass | t/ha | 10 m | int16 | -1 | 16.6 % | 66 | 203 | 411 |
| `grundyta_omdrev2` | Basal area | m²/ha | 10 m | int16 | -1 | 17.0 % | 14 | 36 | 57 |
| `medeldiameter_omdrev2` | Mean stem diameter | cm | 10 m | int16 | -1 | 16.6 % | 18 | 32 | 60 |
| `medelhojd_omdrev2` | Mean tree height | dm | 10 m | int16 | -1 | 16.6 % | 137 | 241 | 402 |
| `p95_omdrev2` | 95th-percentile canopy height | dm | 10 m | int16 | -1 | 13.5 % | 138 | 246 | 413 |
| `vegetationskvot_omdrev2` | Vegetation ratio | % | 10 m | int16 | -1 | 13.9 % | 56 | 96 | 100 |
| `volym_omdrev2` | Stem volume | m³/ha | 10 m | int16 | -1 | 17.0 % | 88 | 398 | 954 |

### Skogliga grunddata — cycle 1 (omdrev 1, 2008–2016, 12.5 m)

Same variables as omdrev 2, but from the older inventory cycle. Native
resolution is 12.5 m; the reference grid for this pipeline matches it, so
no resampling is applied.

| File | Variable | Units | Res. | Dtype | Nodata | zeros% | p50 | p95 | max |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `biomassa_omdrev1` | Above-ground biomass | t/ha | 12.5 m | int16 | -1 | 19.3 % | 78 | 212 | 433 |
| `grundyta_omdrev1` | Basal area | m²/ha | 12.5 m | int16 | -1 | 19.3 % | 18 | 37 | 60 |
| `medeldiameter_omdrev1` | Mean stem diameter | cm | 12.5 m | int16 | -1 | 19.3 % | 20 | 34 | 57 |
| `medelhojd_omdrev1` | Mean tree height | dm | 12.5 m | int16 | -1 | 19.3 % | 151 | 243 | 335 |
| `p95_omdrev1` | 95th-percentile canopy height | dm | 12.5 m | int16 | -1 | 19.2 % | 150 | 250 | 340 |
| `vegetationskvot_omdrev1` | Vegetation ratio | % | 12.5 m | int16 | -1 | 14.8 % | 54 | 95 | 100 |
| `volym_omdrev1` | Stem volume | m³/ha | 12.5 m | int16 | -1 | 19.3 % | 123 | 384 | 939 |

The **~3 pp difference in zeros%** between cycles reflects real change
between the two inventories (primarily clear-cuts / regrowth) plus small
differences in the input lidar point densities.

### SLU Skogskarta — species-resolved layers (12.5 m)

SLU's national forest map, which splits total volume into species-specific
volumes. Useful for mixed-species modelling.

| File | Variable | Units | Res. | Dtype | Nodata | zeros% | p50 | p95 | max |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `slu_skogskarta_biomassa` | Total above-ground biomass | t/ha | 12.5 m | int16 | -1 | 27.2 % | 78 | 204 | 482 |
| `slu_skogskarta_grundyta` | Total basal area | m²/ha | 12.5 m | int16 | -1 | 27.2 % | 19 | 38 | 97 |
| `slu_skogskarta_medeldiameter` | Mean stem diameter | cm | 12.5 m | int16 | -1 | 27.2 % | 17 | 29 | 46 |
| `slu_skogskarta_volym` | Total stem volume | m³/ha | 12.5 m | int16 | -1 | 27.2 % | 121 | 403 | 1 043 |
| `slu_skogskarta_gran_volym` | Norway spruce volume | m³/ha | 12.5 m | int16 | -1 | 29.4 % | 30 | 312 | 948 |
| `slu_skogskarta_tall_volym` | Scots pine volume | m³/ha | 12.5 m | int16 | -1 | 37.7 % | 7 | 159 | 699 |
| `slu_skogskarta_bjork_volym` | Birch volume | m³/ha | 12.5 m | int16 | -1 | 35.1 % | 6 | 65 | 346 |
| `slu_skogskarta_ek_volym` | Oak volume | m³/ha | 12.5 m | int16 | -1 | 81.2 % | 0 | 10 | 353 |
| `slu_skogskarta_bok_volym` | Beech volume | m³/ha | 12.5 m | int16 | -1 | 92.1 % | 0 | 4 | 880 |
| `slu_skogskarta_contorta_volym` | Lodgepole pine volume | m³/ha | 12.5 m | int16 | -1 | **100 %** | 0 | 0 | 0 |
| `slu_skogskarta_ovrigt_volym` | "Other species" volume | m³/ha | 12.5 m | int16 | -1 | 80.8 % | 0 | 16 | 405 |

**Note.** `slu_skogskarta_contorta_volym` is **identically zero** across
this AOI — lodgepole pine (Pinus contorta) is absent from northern Skåne. It
is kept in the dataset for dimensional consistency but carries no signal
here. Ek (oak) and bok (beech) are also rare (>80 % zeros), so models that
treat them as features should expect near-constant columns.

### Canopy height (1 m)

| File | Variable | Units | Res. | Dtype | Nodata | zeros% | p50 | p95 | max |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `tradhojd` | Per-pixel canopy height | dm | 1 m | int16 | -1 | 14.4 % | 54 | 221 | 353 |

Raw 1 m canopy height from laser scanning / photogrammetry: the tallest
vegetation echo inside each 1 m cell. Zeros mean "no vegetation" (water,
roads, clearings, bare ground). Because this layer is resampled from 1 m
onto the 12.5 m reference via bilinear interpolation, each output cell is a
local mean over ~156 source pixels — so it effectively behaves as a finer
alternative to `medelhojd_*` when lidar is available. Compared to
`medelhojd_omdrev2` (10 m, median 137 dm) the median here is lower (54 dm)
because `medelhojd` is a forest-stand average that skips non-forest pixels,
whereas `tradhojd` includes them.

### Soil moisture

| File | Variable | Units | Res. | Dtype | Nodata | zeros% | p50 | p95 | max |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `markfuktighet` | Soil moisture (continuous index) | 0–101 | 2 m | uint16 | 65535 | 3.3 % | 52 | 99 | 101 |
| `markfuktighet_klassad` | Soil moisture class | 1 = dry, 2 = mesic, 3 = mesic-moist, 4 = moist | 2 m | uint8 | 7 | 0 % | 1 | 3 | 4 |

Both layers come from SLU's national soil-moisture model. The continuous
index is more informative for regression; the classified version is
categorical (nearest-neighbour resampled). In this AOI the classified map
uses only classes 1–4 (no "wet"/5 class).

### Flow accumulation

| File | Variable | Units | Res. | Dtype | Nodata | zeros% | p50 | p95 | max |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `flodesackumulering` | Upstream drainage accumulation | m³/s (modelled) | 1 m | float32 | -32768 | 0.0 % | 0.01 | 0.71 | 699.5 |

Native resolution is 1 m. The full Swedish dataset is ~22 GB split into
~60 tiles; for this pipeline the **4 tiles overlapping the AOI** are
merged and cropped to a 11 km × 11.5 km mosaic
(`flodesackumulering_aoi.tif`) before being resampled to 12.5 m (bilinear).
Values are highly skewed — a few cells along river channels carry very high
flow, while the vast majority are near zero.

### Datasets excluded from this build

Downloaded but do **not** overlap the chosen AOI:

- **Förändringsbild stormen Johannes (feb 2026)** — tiles cover northing
  ~6 650 000, roughly 400 km north of the AOI.
- **Indikationer skog med naturvärden (sydostboreal region)** — extent
  starts at northing 6 569 901, roughly 320 km north of the AOI.

Downloaded but **deliberately dropped** from the current CSV:

- **`sksDiffMetaPixel_2024_2025.tif`** — a one-year change-detection raster
  (uint8) with `nodata = 0`. In practice `0` is also a valid class value
  ("no change"), which collapsed 24.7 % of the AOI into NaN and dropped
  ~159 000 rows in the merged CSV. Keeping it would require rewriting the
  nodata metadata; it is excluded pending that fix.
- **`sksDiffMetaVektor_2024_2025.gpkg`** — vector polygons of detected
  change; can be joined later as a boolean "inside change polygon" column
  but is not a raster layer.

### Datasets listed on the portal but not yet downloaded

- Årliga förändringsbilder — Röd (red band)
- Årliga förändringsbilder — Mellaninfraröd (SWIR)
- Årliga förändringsbilder — Vegetationsindex (NDVI)
- ÖSI (Översiktlig skogsinventering)

**Coordinate reference system.** All rasters are in **EPSG:3006 — SWEREF99 TM**,
Sweden's national projected CRS (Transverse Mercator, central meridian 15°E,
scale factor 0.9996, false easting 500 000 m, no false northing). Units are
metres. An easting `x ≈ 390 000` therefore lies ~110 km west of 15°E; a
northing `y ≈ 6 210 000` lies ~6 210 km north of the equator (≈ 56°N).

## 2. Area of interest — why a 10 km square?

### The problem

The raw Skogsstyrelsen / SLU rasters together occupy roughly **62 GB** on
disk. Even a single layer is impractical to hold in memory: `markfuktighet`
at 2 m over the 140 × 150 km Blekinge/Skåne tile alone is ~5 billion pixels
(10 GB as uint16), and `flodesackumulering` at 1 m is ~22 GB across 60+
tiles. Loading 29 such layers and writing them to CSV at full resolution
would exceed normal workstation RAM and produce an unloadable CSV.

### The trade-off

A 10 km × 10 km window gives us:

- **A meaningful ecological sample.** Northern Skåne is a mixed
  coniferous/deciduous forest landscape with enough variation in biomass,
  species mix, soil moisture and terrain to exercise the model without
  being so large that coverage becomes homogeneous.
- **A CSV that fits in RAM.** At the 12.5 m reference grid this is
  801 × 801 = **641 601 rows**. With 22 raster columns and 5 string
  indexruta attributes, the raw CSV is ~108 MB; the enriched one is ~134 MB.
  Both load comfortably with `pandas.read_csv`.
- **Fast iteration.** A full pipeline run (29 rasters → CSV → indexruta
  join) takes ~30 s end to end, so we can change crop or layers and see
  results immediately.

### Specifically, for this project

- **SW corner**: 56.395705°N, 14.123307°E (northern Skåne, Osby–Älmhult area)
- **Size**: 10 000 m × 10 000 m
- **Projected bbox (EPSG:3006)**: `445 875, 6 250 462.5 → 455 887.5, 6 260 475`
  (the lat/lon point is reprojected via `pyproj`, then snapped *outward* to
  the reference raster's 12.5 m grid so any pixel the square overlaps is
  included — no partial pixels are dropped at the edges).

### Why 12.5 m as the reference grid?

The script uses the **alphabetically first** raster as the reference grid.
That happens to be `biomassa_omdrev1.tif` at 12.5 m. We could have chosen a
finer reference (10 m for omdrev 2, or 1 m for tradhojd) — but:

- **10 m would only recover** the omdrev-2 native resolution;
  omdrev-1/SLU Skogskarta layers are still 12.5 m native and would be
  up-sampled anyway.
- **1 m would explode the row count** by a factor of 156 (to ~100 M rows)
  without adding information to the 12.5 m-native layers.

12.5 m is the coarsest *native* resolution among the non-high-res layers,
so it is a **lossless** choice for all "Skogliga grunddata omdrev 1" and
"SLU Skogskarta" layers; only omdrev 2 (10 m), `markfuktighet` (2 m),
`tradhojd` (1 m) and `flodesackumulering` (1 m) are down-sampled. For
disaggregated modelling the script accepts a different `--aggregate` factor
or the reference could be renamed to a finer one (or the script modified to
pick the finest raster as reference).

## 3. Raster → CSV conversion

Script: [`rasters_to_csv.py`](rasters_to_csv.py).

Steps, in order:

1. **Enumerate** all `*.tif` in the input folder. The first raster
   (alphabetically) becomes the **reference grid**; its resolution,
   transform and CRS define the output lattice.
2. **Windowed read.** Only the pixels inside the crop bbox are read
   (`rasterio.windows.from_bounds`), so memory usage is proportional to the
   area of interest, not the full raster.
3. **Reproject other rasters onto the reference grid.** When a raster's CRS,
   resolution or alignment differs from the reference it is resampled:
   - **bilinear** for continuous variables (float/int with magnitude meaning),
   - **nearest** for categorical variables (detected via integer dtype).
   In the current build the reference grid is `biomassa_omdrev1` at **12.5 m**
   (alphabetically first). All omdrev-2 layers (10 m) and `markfuktighet`
   (2 m) are resampled onto this 12.5 m lattice.
   Source rasters are also windowed — the crop bbox is transformed into the
   source CRS, expanded by a 10-pixel buffer to avoid reproject-edge
   artefacts, and only that window is read.
4. **Nodata handling.** Each raster's declared `nodata` value is masked to
   `NaN` before resampling and before the final merge.
5. **Optional `N × N` aggregation** (not used for the current dataset) would
   average continuous layers and take the per-block mode of categorical ones.
6. **Assemble the table.** One row per reference-grid cell, columns:
   `x, y, <layer names...>`, with `(x, y)` being the cell **center** in
   EPSG:3006 meters. Rows where *any* layer is nodata are dropped.

Run:

```bash
uv run python Projects/rasters_to_csv.py \
    Projects/combined_rasters Projects/out_10km.csv \
    --sw 56.395705 14.123307            # defaults to --size 10000
```

Output: **`out_10km.csv`**, 801 × 801 = **641 601 rows** (no nodata dropped
over this area), 31 columns (`x, y` + 29 raster layers), **~108 MB** on disk.

## 4. Enrichment — Sverige-indexruta

Script: [`join_indexruta.py`](join_indexruta.py).

Skogsstyrelsen publishes a **500 m × 500 m index grid** ("Sverige
indexruta") as an ArcGIS MapServer layer:

```
https://geodpags.skogsstyrelsen.se/arcgis/rest/services/
  Geodataportal/GeodataportalVisaSverigeindexruta/MapServer/0
```

The `.kmz` file in the repo
(`Geodataportal_GeodataportalVisaSverigeindexruta.kmz`) is only a Google
Earth **NetworkLink** to the server's PNG-tile renderer — it contains no
vector geometry. To get per-cell attributes we therefore call the server's
GeoJSON `query` endpoint with the AOI bbox in EPSG:3006. This returns 441
features covering our 10 km square.

Each feature carries useful metadata. We keep:

| Attribute | Description |
|---|---|
| `BK` | Unique block code, e.g. `6250_453_05` |
| `PageName` | Map page name, e.g. `IG383` |
| `Storruta` | Enclosing larger block, e.g. `625_45` |
| `CenterLanNamn` | County (Län), e.g. `SKÅNE LÄN` |
| `CenterKommunNamn` | Municipality, e.g. `OSBY` |

**Join strategy.** Because the index grid is axis-aligned to SWEREF99 TM at
500 m spacing, every feature's `SWEREF99Ost`/`SWEREF99Nord` pair is the
**cell center** (corner + 250 m). We can therefore skip a full spatial
point-in-polygon operation: for each CSV row we compute the enclosing
cell's center key

```
cx = (x // 500) * 500 + 250
cy = (y // 500) * 500 + 250
```

and look it up in a dict built from the fetched features. This runs in
O(rows) with no geopandas/shapely dependency and matches 100 % of rows for
the current AOI.

Run:

```bash
uv run python Projects/join_indexruta.py \
    Projects/out_10km.csv Projects/out_10km_idx.csv
```

Output: **`out_10km_idx.csv`**, identical row count, with the five attribute
columns appended, **~134 MB** on disk. 0 unmatched rows.

## 5. Final columns

`x, y, biomassa_omdrev1, biomassa_omdrev2, flodesackumulering, grundyta_omdrev1, grundyta_omdrev2, markfuktighet, markfuktighet_klassad, medeldiameter_omdrev1, medeldiameter_omdrev2, medelhojd_omdrev1, medelhojd_omdrev2, p95_omdrev1, p95_omdrev2, slu_skogskarta_biomassa, slu_skogskarta_bjork_volym, slu_skogskarta_bok_volym, slu_skogskarta_contorta_volym, slu_skogskarta_ek_volym, slu_skogskarta_gran_volym, slu_skogskarta_grundyta, slu_skogskarta_medeldiameter, slu_skogskarta_ovrigt_volym, slu_skogskarta_tall_volym, slu_skogskarta_volym, tradhojd, vegetationskvot_omdrev1, vegetationskvot_omdrev2, volym_omdrev1, volym_omdrev2, BK, PageName, Storruta, CenterLanNamn, CenterKommunNamn`

`x` and `y` are cell centers in EPSG:3006 metres. To map a row back to
lat/lon:

```python
from pyproj import Transformer
t = Transformer.from_crs(3006, 4326, always_xy=True)
lon, lat = t.transform(x, y)
```

## 6. Environment

Managed with `uv`; `pyproject.toml` at the repo root pins Python ≥ 3.12 and
declares `rasterio`, `pyproj`, `numpy`, `pandas`, `scipy`, plus the modelling
libraries (`pyro-ppl`, `torch`, `scikit-learn`, ...). Recreate with:

```bash
uv sync
```

## 7. Reproducibility checklist

- Raw GeoTIFFs are immutable inputs (`combined_rasters/`).
- `rasters_to_csv.py` is deterministic given the same `--sw` / `--size` /
  `--aggregate` arguments and the same reference raster (alphabetical first:
  `biomassa_omdrev1.tif`).
- `join_indexruta.py` queries a live ArcGIS service; the index grid is
  static, but the query date can be recorded if strict reproducibility is
  required.
- All CRS conversions go through `pyproj` / `rasterio`'s built-in PROJ
  bindings — no hand-rolled projection math.
