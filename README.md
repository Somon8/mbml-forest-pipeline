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

- **Browser:** [Releases → v1.0-data](https://github.com/Somon8/mbml-forest-pipeline/releases/tag/v1.0-data) → scroll to *Assets* → click `out_10km_idx.csv`.
- **curl:**
  ```bash
  curl -LO https://github.com/Somon8/mbml-forest-pipeline/releases/download/v1.0-data/out_10km_idx.csv
  ```
- **gh CLI:**
  ```bash
  gh release download v1.0-data -R Somon8/mbml-forest-pipeline -p out_10km_idx.csv
  ```

Assets on that release:

| File | Size | Contents |
|---|---|---|
| `out_10km_idx.csv` | 134 MB | 29 raster layers + indexruta columns (BK, PageName, Storruta, CenterLanNamn, CenterKommunNamn) — **use this one** |
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
