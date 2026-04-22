# Project journal — MBML forest modelling (42186)

Running log of decisions, findings and open items for the course project.
Newest entries go at the top; every entry gets a `## YYYY-MM-DD — title`
header. Think of it as a lab notebook that feeds directly into the report.

**Conventions**

- One dated section per working session.
- Each section is self-contained: context, what was done, what the data
  showed, decisions, artefacts produced, open items.
- Cross-reference files with relative paths (e.g. `Preprocessing.ipynb`)
  so they resolve as links in the IDE.

---

## 2026-04-22 — Spatial features (neighbourhood / directional / distance)

Added a per-pixel spatial-context feature step on top of the preprocessed
CSV, motivated by the observation that neighbouring cells carry information
the individual pixel does not — most concretely, taller trees to the south
of a pixel shade it (at 56° N the sun never crosses into the northern sky),
so a "south-minus-north height" term is a direct illumination proxy. The
same machinery also gives canopy roughness, forest-edge proximity, and
distance-to-water without pulling in additional data.

Artefact: **`spatial_features.py`**. Runs on `out_10km_idx_preprocessed.csv`
in place by default; idempotent (drops its own output columns before
recomputing). Takes ~10 s end to end.

### Design

- The preprocessed CSV is a complete 801 × 801 grid at 12.5 m, so the
  target layer and the two boolean masks pivot directly to 2D arrays with
  row 0 = northernmost y, column 0 = westernmost x. Everything else is
  `scipy.ndimage`:
  - `uniform_filter` for 3 × 3 / 5 × 5 mean and std (via
    `E[X²] − E[X]²`, clipped at 0 to absorb float error)
  - `convolve` with four hand-built directional kernels for N/E/S/W means
  - `sobel` for the gradient (row-axis sign flipped because row 0 is north)
  - `distance_transform_edt` on the inverse mask, scaled by 12.5 m
- Boundaries: `reflect` everywhere. AOI-edge cells therefore get
  plausible-but-fabricated neighbours rather than NaN; acceptable given
  the 10 km AOI vs. a 5-cell kernel radius.
- Target layer: **`p95_omdrev2`** (10 m 95th-percentile canopy height). It
  already passed §2.4 / §2.8 of the previous session as the cleanest height
  proxy and is on the 12.5 m reference grid after resampling.

### Target layers and columns added (18)

Eight per-layer features are computed for every layer in the script's
`TARGET_LAYERS` list. The list currently holds both height candidates —
**`p95_omdrev2`** (95th-percentile canopy) and **`medelhojd_omdrev2`**
(stand mean) — because either could end up as the modelling target y, and
the spatial-context features are target-specific. Building both now avoids
a second pass and the two sets come out near-identical (per §2.8 the
layers track each other on forested pixels), so the choice is deferred
to the modelling step. Plus two target-agnostic distance features.

**Per-layer (×2 layers = 16 columns):**

| Column | Meaning |
|---|---|
| `<L>_mean3`, `_std3` | 3 × 3 neighbourhood mean and std (dm) |
| `<L>_mean5`, `_std5` | 5 × 5 neighbourhood mean and std (dm) |
| `<L>_dNS` | mean(S 3 cells) − mean(N 3 cells), dm. Positive ⇒ taller south ⇒ pixel shaded from south |
| `<L>_dEW` | mean(E) − mean(W), dm |
| `<L>_grad_mag` | Sobel gradient magnitude (dm / cell) |
| `<L>_grad_aspect` | `atan2(dZ/dy_geo, dZ/dx_geo)`, rad (0 = east, π/2 = north) |

**Target-agnostic (2 columns):**

| Column | Meaning |
|---|---|
| `dist_to_no_forest_m` | Euclidean distance to nearest `is_no_forest` pixel, m |
| `dist_to_lake_m` | Euclidean distance to nearest `is_lake` pixel, m |

Using `is_no_forest` and `is_lake` (both materialised in
`Preprocessing.ipynb`) for the distance features means the edge / water
definitions agree exactly with the mask logic from the previous session:
non-forest in *both* inventory cycles for the former, `markfuktighet_klassad
== 4` for the latter.

### Summary stats on the current AOI

```
                               p50       p95       max      unit
p95_omdrev2_mean3             131.6     233.6     324.2     dm
p95_omdrev2_std3               23.4      82.5     149.6     dm
p95_omdrev2_mean5             129.4     226.7     303.1     dm
p95_omdrev2_std5               37.8      87.3     139.5     dm
p95_omdrev2_dNS                 0.0      84.0     303.3     dm
p95_omdrev2_dEW                 0.0      86.7     299.7     dm
p95_omdrev2_grad_mag          150.9     661.5    1338.0     dm / cell
medelhojd_omdrev2_mean3       130.1     228.9     317.2     dm
medelhojd_omdrev2_std3         23.2      81.7     146.2     dm
medelhojd_omdrev2_mean5       127.9     222.3     296.7     dm
medelhojd_omdrev2_std5         37.2      86.5     137.3     dm
medelhojd_omdrev2_dNS           0.0      82.7     296.7     dm
medelhojd_omdrev2_dEW           0.0      85.3     293.0     dm
medelhojd_omdrev2_grad_mag    149.6     655.1    1305.0     dm / cell
dist_to_no_forest_m            72.9     234.9     499.5     m
dist_to_lake_m                600.0    1307.0    1765.1     m
```

`mean3` median 131.6 dm (p95) / 130.1 dm (medelhojd) matches the raw-layer
medians (138 / 137 dm) to within a few dm — the expected sanity check that
a 3 × 3 mean of a slowly-varying layer tracks the point value. `dNS` and
`dEW` medians are exactly 0 for both targets, so the directional signal is
symmetric in bulk and only the tails carry information — exactly what you
want from a differential feature (no additive offset contaminating the
"flat" case). The two targets produce nearly identical distributions
across every feature, consistent with their high forest-pixel correlation
(§2.8); the choice between them is therefore a modelling decision, not a
spatial-feature one.

### Implications / next steps

- Redundancy with existing covariates: `mean3 / mean5` correlate strongly
  with `p95_omdrev2` itself and with each other. For a linear model keep
  at most one; for a tree-based model keep both and let the splits sort
  it out.
- The **dNS** feature is the one explicitly hypothesised about
  (south-illumination). If it comes out significant in the model, that is
  a narratable sentence for the report: "taller canopy immediately south
  of a pixel predicts reduced [growth / biomass / whatever] in that
  pixel, consistent with shading at 56° N". If it does not, that is also
  worth reporting — the Swedish lidar resolution may be too coarse for
  single-tree shading to be resolvable at 12.5 m.
- Not added deliberately: the same stats on `biomassa_omdrev2` /
  `volym_omdrev2` / `tradhojd`. Given the cross-layer correlations from
  §2.8 they would be near-duplicates of the `p95` / `medelhojd` versions.
  Easy to extend though — just append the layer name to the script's
  `TARGET_LAYERS` list.
- `flodesackumulering`-based distance (nearest high-flow cell ≈ stream
  proxy) is a candidate for later if a hydrology angle is wanted.

---

## 2026-04-22 — Exploratory analysis and preprocessing masks

Exploratory analysis and preprocessing design for the Swedish forest tabular
dataset (`out_10km_idx.csv`) ahead of Bayesian modelling in Pyro.

Inputs assumed:

- `out_10km_idx.csv` — 641 601 rows × 36 cols (29 raster layers + `x, y` +
  5 indexruta attributes), produced by `rasters_to_csv.py` +
  `join_indexruta.py` over the 10 km × 10 km AOI in northern Skåne described
  in `DATA_PIPELINE.md`.
- 801 × 801 regular grid at 12.5 m cell size (EPSG:3006). No missing values.

Artefacts produced today:

- **`Exploratory analysis.ipynb`** — end-to-end EDA, extended with a
  "Preprocessing masks — before / after" section.
- **`Preprocessing.ipynb`** — new notebook that materialises the boolean
  mask columns and writes `out_10km_idx_preprocessed.csv`.
- **`PROJECT_JOURNAL.md`** — this document (first entry).

---

### 1. Dataset recap (for the report)

Geographic footprint: 10 km × 10 km square, SW corner at
`56.395705°N, 14.123307°E` (northern Skåne, Osby / Östra Göinge area).
Everything is on the 12.5 m reference grid (`biomassa_omdrev1.tif`), so
no resampling is required for the omdrev 1 or SLU Skogskarta layers;
omdrev 2 (10 m), `markfuktighet` (2 m), `tradhojd` (1 m) and
`flodesackumulering` (1 m) are down-sampled.

Column groupings used throughout (codified in the EDA notebook):

| Group | Columns |
|---|---|
| `OMDREV2` | `biomassa_omdrev2, grundyta_omdrev2, medeldiameter_omdrev2, medelhojd_omdrev2, p95_omdrev2, vegetationskvot_omdrev2, volym_omdrev2` |
| `OMDREV1` | same 7 variables, `_omdrev1` suffix |
| `SLU_TOTALS` | `slu_skogskarta_biomassa, _grundyta, _medeldiameter, _volym` |
| `SLU_SPECIES` | `slu_skogskarta_gran_volym, _tall_volym, _bjork_volym, _ek_volym, _bok_volym, _contorta_volym, _ovrigt_volym` |
| `MOISTURE` | `markfuktighet, markfuktighet_klassad` |
| `OTHER` | `tradhojd, flodesackumulering` |
| `ADMIN` | `BK, PageName, Storruta, CenterLanNamn, CenterKommunNamn` |

Unit conventions (from `DATA_PIPELINE.md`, repeated here because they
matter for any plot / prior): heights in **decimeters**, diameter in **cm**,
biomass in **t/ha**, volume in **m³/ha**, basal area in **m²/ha**,
vegetation ratio in **%**, `markfuktighet` is a 0–101 continuous index,
`markfuktighet_klassad` is ordinal (1 dry, 2 mesic, 3 mesic-moist, 4 moist).

---

### 2. What the EDA showed

#### 2.1 Integrity

- **0 missing values** across all 641 601 rows and 36 columns. The pipeline's
  nodata-masking + drop step in `rasters_to_csv.py` is verified to be
  consistent with the documented per-layer nodata sentinels.
- Reproduced `DATA_PIPELINE.md`'s zeros% column exactly — CSV matches the
  raster-level statistics, no silent corruption from resampling / join.

#### 2.2 Zero-inflation (the dominant structural property)

Fraction of pixels equal to exactly 0 per layer:

| Layer | zeros % |
|---|---:|
| `tradhojd` | 14.4 |
| `vegetationskvot_omdrev2` | 13.9 |
| `p95_omdrev2` | 13.5 |
| `biomassa_omdrev2`, `medelhojd_omdrev2`, `medeldiameter_omdrev2` | 16.6 |
| `grundyta_omdrev2`, `volym_omdrev2` | 17.0 |
| `biomassa_omdrev1` and cycle-1 siblings | 19.2–19.3 |
| `slu_skogskarta_{biomassa,grundyta,volym,medeldiameter}` | 27.2 |
| `slu_skogskarta_gran_volym` | 29.4 |
| `slu_skogskarta_bjork_volym` | 35.1 |
| `slu_skogskarta_tall_volym` | 37.7 |
| `slu_skogskarta_ovrigt_volym` | 80.8 |
| `slu_skogskarta_ek_volym` | 81.2 |
| `slu_skogskarta_bok_volym` | 92.1 |
| `slu_skogskarta_contorta_volym` | **100.0** |

Every forest-measuring layer is zero-inflated. A naïve Gaussian likelihood
will be pulled toward zero and miscalibrate the positive-value residuals.

#### 2.3 Cycle 1 vs cycle 2 consistency

Hexbin plots of `(*_omdrev1, *_omdrev2)` for biomass, volume, height, basal
area show:

- Strong linear alignment around the 1:1 line — the two cycles measure the
  same physics; they're **redundant** for a snapshot model.
- A heavy mass at exactly (0, 0) (non-forest in both cycles).
- An off-diagonal ridge where cycle 2 > cycle 1 (natural growth over ~10 yr).
- A thinner, negative off-diagonal (clear-cuts / thinning / storm damage).
- The ~3 pp zeros% difference between cycles (omdrev 1 has more zeros) is
  consistent with `DATA_PIPELINE.md`'s note that this reflects real change
  (clear-cuts / regrowth) plus small lidar-density differences.

#### 2.4 Height layers

- `medelhojd_omdrev2` is a **stand average** over forested pixels only —
  median 137 dm (≈13.7 m) over forest pixels.
- `tradhojd` is a **per-pixel canopy height** including bare ground as 0 —
  median 54 dm (≈5.4 m) over *all* pixels.
- On forested pixels, the two agree on a ~1:1 line with modest scatter.
  `tradhojd` is effectively a higher-resolution re-expression of the same
  quantity once we mask non-forest out.

#### 2.5 Species composition (SLU Skogskarta)

Confirmed qualitatively:

- `slu_skogskarta_contorta_volym` is identically zero over the AOI — drop
  it from any model.
- Oak (`ek`) and beech (`bok`) exceed 80 % zeros; treat as rare with a
  shrinkage prior if kept.
- `sum(species volumes) ≈ slu_skogskarta_volym` across the AOI — species
  decomposition is internally consistent.

#### 2.6 Soil moisture

- `markfuktighet` (continuous) and `markfuktighet_klassad` (ordinal) are
  two views of the same model; continuous is cleaner for regression,
  ordinal better for grouping.
- In this AOI the classified map uses only classes 1–4 (no "wet" / 5
  class).
- **Local observation for the report:** class 4 in this AOI maps to
  standing water (lakes), not the nominal label "moist". Basis: visual
  inspection on the spatial maps — class-4 pixels form compact, hydrologically
  plausible polygons coincident with known lakes. Documented in the
  preprocessing notebook and used to build `is_lake`.

#### 2.7 Flow accumulation — extreme skew

`flodesackumulering` has p50 ≈ 0.01 and max ≈ 699.5 — ~5 orders of magnitude.
Raw histogram is useless; `log1p(flodesackumulering)` is the form used
everywhere else in the notebook and should be the form fed to any model.

#### 2.8 Correlation structure (forest pixels only)

Computed on `medelhojd_omdrev2 > 0` to avoid inflation from the shared zero
mass. High-level findings:

- Within cycle 2, `biomassa ≈ volym ≈ grundyta ≈ basal-area block`
  correlations are very high (ρ > 0.9). **Redundancy alert** — do not put
  all of them into a regression as independent features.
- `medelhojd` and `p95` are near-duplicates of each other (both height
  summaries from the same lidar).
- `markfuktighet` decouples from the forest block — it carries independent
  signal.
- `tradhojd` aligns tightly with `medelhojd_omdrev2` on forested pixels.

#### 2.9 Spatial structure

The 801 × 801 grid pivots directly into a 2D image without any GIS library.
Visual inspection shows:

- Legible forest stands, clear-cut blocks, roads, and lakes.
- Species maps show distinct gran (spruce) vs tall (pine) stands, with
  birch mostly as filler and oak/beech only at a few patches.
- `flodesackumulering` (log1p) resolves the drainage network as thin
  bright lines cutting through a near-zero background.
- Kommun boundary is visible in the administrative map: OSBY dominates
  (525 681 cells), ÖSTRA GÖINGE the remainder (115 920).

#### 2.10 Administrative context

- `CenterLanNamn`: 1 unique (SKÅNE LÄN across the entire AOI) — not useful
  as a grouping variable here.
- `CenterKommunNamn`: 2 unique (OSBY, ÖSTRA GÖINGE) — usable but low-cardinality.
- `Storruta`: 4 unique — usable.
- `BK` / `PageName`: 441 unique 500 m indexruta cells — **the natural
  hierarchical grouping variable.** Each cell contains exactly 1600 rows
  (40 × 40 pixels at 12.5 m), so partial pooling across indexruta is cheap
  and well-specified.

#### 2.11 Forest-vs-non-forest mask consistency

Agreement of `(column > 0) == (volym_omdrev2 > 0)` over all 641 601 pixels:

| Column | agreement |
|---|---:|
| `slu_skogskarta_*` (all four totals) | 0.8349 |
| `tradhojd` | 0.8370 |
| `p95_omdrev2` | 0.9645 |
| `vegetationskvot_omdrev2` | 0.9685 |
| `biomassa_omdrev2`, `medeldiameter_omdrev2`, `medelhojd_omdrev2` | 0.9958 |
| `grundyta_omdrev2` | 1.0000 |
| `volym_omdrev2` | 1.0000 |

Within cycle 2 Skogliga grunddata the forest/non-forest classification is
almost perfectly consistent. SLU Skogskarta and `tradhojd` disagree on
~16 % of pixels — expected, since they come from different products with
different forest-edge thresholds. This justified using **cycle-2 Skogliga
grunddata** as the canonical source for the `is_no_forest` mask.

---

### 3. Preprocessing — decisions and mask definitions

Objective: add boolean columns that downstream modelling notebooks can use
to filter pixels, without modifying any raw layer. Output:
`out_10km_idx_preprocessed.csv`.

#### 3.1 Exclusion masks

| Mask | Rule | Rationale |
|---|---|---|
| `is_no_forest` | `medelhojd_omdrev1 == 0 AND medelhojd_omdrev2 == 0` | Pixel is non-forest in *both* cycles — avoids dropping transient clear-cuts that regrew, or vice-versa. Uses `medelhojd` since it is the stand-average height from Skogsstyrelsen and agrees 99.58 % with `volym_omdrev2 > 0`. |
| `is_lake` | `markfuktighet_klassad == 4` | Class 4 in this AOI empirically corresponds to lakes (see §2.6). |

#### 3.2 Disturbance masks (cycle 2 − cycle 1 < 0)

One per forest metric. Because heights / volumes / biomass are
non-negative, `cycle2 − cycle1 < 0` can only fire when cycle 1 was
positive, so pixels that were non-forest in both cycles are correctly not
flagged.

| Mask | Underlying variable (unit) |
|---|---|
| `delta_neg_medelhojd` | `medelhojd_omdrev{1,2}` (dm) |
| `delta_neg_p95` | `p95_omdrev{1,2}` (dm) |
| `delta_neg_medeldiameter` | `medeldiameter_omdrev{1,2}` (cm) |
| `delta_neg_biomassa` | `biomassa_omdrev{1,2}` (t/ha) |
| `delta_neg_volym` | `volym_omdrev{1,2}` (m³/ha) |

The spatial maps of these masks light up in the same blocks (clear-cuts,
storm patches, thinning rides), and the 5 masks strongly co-occur per
pixel — this is the basis for the combined mask below.

#### 3.3 Combined mask

```
is_stable_forest = ~is_no_forest
                 & ~is_lake
                 & ~any(delta_neg_*)
```

Interpretation: "pixels that were forest in both cycles, are not water,
and did not lose stem height / basal area / volume / biomass between
cycles." This is the default mask for any steady-state forest model. The
primitive masks are kept as separate columns so a looser rule (e.g.
`~is_no_forest & ~is_lake` only) can still be assembled downstream.

#### 3.4 Why these rules and not alternatives

- **Both cycles** required in `is_no_forest`: an intersection, not union.
  Using union would drop transiently clear-cut pixels that regenerated —
  those are exactly the pixels we want to study for disturbance / growth
  dynamics, not throw away.
- **`medelhojd` as the forest marker** and not `volym_omdrev2`: both give
  the same answer to 4 decimal places (agreement 0.9958 vs 1.0000) but
  `medelhojd` is present in both cycles, which is what the rule needs.
- **Strict `< 0`** for the delta masks: heights are integer decimeters and
  biomass/volume are integer t/ha / m³/ha, so any non-zero negative delta
  already corresponds to real change beyond the nearest-integer
  measurement floor. Adding a tolerance would swallow genuine small
  losses.
- **`is_stable_forest` via union of disturbance masks** (not just the
  strongest one): the 5 metrics disagree on edge cases near zero, and an
  OR is the conservative choice — if any dimension lost signal, we call
  the pixel disturbed.

---

### 4. Effect of masks (before / after plots)

Added to the EDA notebook in the "Preprocessing masks — before / after"
section:

1. **Exclusion overlay on a biomass map** — `is_no_forest` and `is_lake`
   clearly sit on the non-forest zero regions and the visible lake
   polygons. No visible leakage into forest pixels.
2. **Biomass map with masked pixels set to NaN** — colour scale resolves
   forest-internal variation that was previously crushed by the
   non-forest zeros.
3. **Five delta histograms + summary panel** — the `k of 5` disturbance
   count has most pixels at 0 (no disturbance), a secondary mode at 5
   (strong clear-cuts hit all metrics together), and few intermediate
   values — indicating the 5 masks are correlated, not complementary.
4. **Spatial map of disturbance count per pixel** — highlights coherent
   clear-cut patches rather than scattered noise, which is the expected
   sign that the masks are catching real disturbance not measurement
   flicker.
5. **Per-variable before/after distributions** (2 × 3 grid) — for each of
   `medelhojd`, `p95`, `medeldiameter`, `biomassa`, `volym`, applying
   `is_stable_forest` removes the zero spike cleanly while leaving the
   positive forest distribution's shape intact. This is the direct
   evidence that the combined mask is doing the right thing for any of
   the five candidate target variables.
6. **`is_stable_forest` spatial map** — visually similar to a thresholded
   biomass map with lakes punched out, which is what we want.

---

### 5. Implications for Bayesian modelling

Collected from the EDA for use in the report's methodology section:

1. **Likelihood structure.** Every candidate target is zero-inflated.
   Use a hurdle / zero-inflated likelihood, e.g.
   `Bernoulli(is_forest) × LogNormal(μ, σ)` on the positive part, rather
   than a single Gaussian.
2. **Pre-filtering vs likelihood modelling.** Two valid routes:
   (a) use `is_stable_forest` as a filter and model only the positive
   distribution (cleaner but loses change information);
   (b) model the zero-inflation explicitly with the raw data and predict
   `is_no_forest` from covariates. `Preprocessing.ipynb` keeps both
   options open by materialising the masks as columns rather than
   rewriting rows.
3. **Cycle redundancy.** `*_omdrev1` and `*_omdrev2` are near-duplicates.
   Pick one for a snapshot model (omdrev 2 is newer and finer-resolution).
   Use both only when modelling change.
4. **Feature redundancy inside a cycle.** Within one cycle, do not feed
   `biomassa / grundyta / volym / medelhojd / p95` all in as independent
   predictors — ρ > 0.9 pairwise. Pick 1–2.
5. **Species handling.** Drop `contorta` (identically zero). For rare
   species (`ek`, `bok`, `ovrigt`) either drop or apply a shrinkage prior
   with a species-level hierarchy.
6. **Transforms.** `flodesackumulering` → `log1p`. Heights are already in
   dm but report in m for readability. Volumes / biomass are heavy-tailed
   — consider `log1p` for them too in any linear model.
7. **Grouping variables for hierarchical models.** `BK` / `PageName`
   (441 indexruta cells, 1600 pixels each) is the natural partial-pooling
   level. `CenterKommunNamn` (2 levels) is too coarse to partially pool
   usefully; `CenterLanNamn` (1 level) contains no group information.
8. **Spatial priors.** Grid is complete and regular, so a GP on
   `(x, y)` or a CAR / ICAR prior on indexruta is straightforward. No
   coverage gaps to worry about.
9. **Disturbance as a target.** If the report wants a change-detection
   model, the disturbance masks are ready-made binary labels per pixel.

---

### 6. Open items (for next session)

- Decide on the primary modelling target (candidates: `volym_omdrev2`,
  `biomassa_omdrev2`, `medelhojd_omdrev2`). Pick based on which aligns
  with the report's framing (growth modelling vs. biomass estimation vs.
  something else).
- Decide between masking (`is_stable_forest`) and likelihood-modelling
  the zero-inflation.
- Pick the feature subset — current suggestion: 1 forest target + 1
  feature from each non-redundant block (`slu_skogskarta_gran_volym`,
  `slu_skogskarta_tall_volym`, `markfuktighet`, `log1p(flodesackumulering)`,
  `tradhojd` optional).
- Check whether `CenterKommunNamn` adds anything over indexruta-level
  pooling, or whether the 2-level split is redundant.
- Still downloadable but not yet in the AOI: annual NDVI / SWIR / red-band
  change rasters, and ÖSI — could be pulled in if a richer predictor set
  is wanted.
