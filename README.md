# mbml-forest-pipeline

Pipeline for turning Swedish forest raster data (Skogsstyrelsen / SLU) into
a flat tabular dataset suitable for Bayesian / probabilistic modelling.
Built for the DTU course **42186 Model-based Machine Learning**.

See **[DATA_PIPELINE.md](DATA_PIPELINE.md)** for the full description of
input layers, coordinate system, crop workflow, per-layer statistics and
reproducibility notes.

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
