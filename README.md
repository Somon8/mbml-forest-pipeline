# mbml-forest-pipeline

Pipeline for turning Swedish forest raster data (Skogsstyrelsen / SLU) into
a flat tabular dataset suitable for Bayesian / probabilistic modelling.
Built for the DTU course **42186 Model-based Machine Learning**.

See **[DATA_PIPELINE.md](DATA_PIPELINE.md)** for the full description of
input layers, coordinate system, crop workflow, per-layer statistics and
reproducibility notes.

## Download the prepared dataset

The 10 km × 10 km AOI (SW corner `56.395705, 14.123307`, 641,601 rows, 29
raster layers + Sverige-indexruta attributes) is published as a GitHub
Release — no need to re-run the pipeline or download the raw rasters.

### Preprocessed + spatial features (recommended for modelling) — v1.2-data

- **Browser:** [Releases → v1.2-data](https://github.com/Somon8/mbml-forest-pipeline/releases/tag/v1.2-data) → scroll to *Assets* → click `out_10km_idx_preprocessed.csv`.
- **curl:**
  ```bash
  curl -LO https://github.com/Somon8/mbml-forest-pipeline/releases/download/v1.2-data/out_10km_idx_preprocessed.csv
  ```
- **gh CLI:**
  ```bash
  gh release download v1.2-data -R Somon8/mbml-forest-pipeline -p out_10km_idx_preprocessed.csv
  ```

| File | Size | Contents |
|---|---|---|
| `out_10km_idx_preprocessed.csv` | 351 MB | Preprocessed CSV **plus** 18 per-pixel spatial features (neighbourhood mean / std, N–S and E–W directional differentials, Sobel gradient, and distance-to-`is_no_forest` / `is_lake`) computed for both `p95_omdrev2` and `medelhojd_omdrev2`. Produced by [`Preprocessing.ipynb`](Preprocessing.ipynb) then [`spatial_features.py`](spatial_features.py). See [DATA_PIPELINE.md §6](DATA_PIPELINE.md#6-spatial-features--neighbourhood-directional-distance-to-mask) for details. |

### Previous version — v1.1-data (preprocessed only, no spatial features)

Kept for reproducibility of earlier work. Same 641,601 rows as v1.2 but
without the 18 spatial-feature columns ([v1.1-data release](https://github.com/Somon8/mbml-forest-pipeline/releases/tag/v1.1-data)).

| File | Size | Contents |
|---|---|---|
| `out_10km_idx_preprocessed.csv` | 162 MB | Cleaned + feature-engineered version of `out_10km_idx.csv`. See [`Preprocessing.ipynb`](Preprocessing.ipynb) for the exact steps. |

### Raw flat dataset — v1.0-data

If you want to redo preprocessing yourself, the raw flat dataset is on [v1.0-data](https://github.com/Somon8/mbml-forest-pipeline/releases/tag/v1.0-data):

| File | Size | Contents |
|---|---|---|
| `out_10km_idx.csv` | 134 MB | 29 raster layers + indexruta columns (BK, PageName, Storruta, CenterLanNamn, CenterKommunNamn) |
| `out_10km.csv` | 108 MB | Raster layers only, no indexruta join |

## Quick start

```bash
uv sync

# 1. Put the raw GeoTIFFs somewhere on disk and symlink them into
#    combined_rasters/ (see DATA_PIPELINE.md §1 for which layers to grab).
#
#    Example:
#    ln -s /path/to/sks_Biomassa_omdrev2_12/sksBiomassa12.tif \
#          combined_rasters/biomassa_omdrev2.tif

# 2. Pick a SW corner in Google Maps (right-click → copy lat,lon) and
#    extract a 10 km square cropped to the reference raster grid:
uv run python rasters_to_csv.py combined_rasters out_10km.csv \
    --sw 56.395705 14.123307

# 3. Enrich with Sverige-indexruta metadata (Län / Kommun / block codes):
uv run python join_indexruta.py out_10km.csv out_10km_idx.csv
```

## Scripts

| File | Purpose |
|---|---|
| [`rasters_to_csv.py`](rasters_to_csv.py) | Crop + reproject + merge a folder of GeoTIFFs into one CSV |
| [`join_indexruta.py`](join_indexruta.py) | Attach Skogsstyrelsen 500 m index-grid attributes to every row |
| [`DATA_PIPELINE.md`](DATA_PIPELINE.md) | Full data description and design notes |
