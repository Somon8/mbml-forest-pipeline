# Claude Code prompt — model comparison notebook (DTU 42186)

Copy everything between the `===` lines into Claude Code, run from
`/Users/mondrup/Library/Mobile Documents/com~apple~CloudDocs/DTU/Model-based machine learning 42186/Projects`.

===

## Task

Build a model comparison notebook for the DTU 42186 Model-based Machine Learning project. The framework predicts forest **volume growth** between two Skogsstyrelsen inventory cycles (omdrev1 → omdrev2) on a Swedish forest pixel dataset, and compares seven models across point metrics, probabilistic metrics, runtime, and scaling behaviour. The deliverable is a single reproducible notebook with a `models/` subfolder of pluggable model implementations, plus cached intermediate results.

Read `Projects/PROJECT_JOURNAL.md` and `Projects/DATA_PIPELINE.md` before starting — both contain critical context (zero-inflation patterns, mask definitions, indexruta grouping, unit conventions). Re-read them whenever you make a non-trivial design choice.

## Inputs

- Data file: `Projects/out_10km_idx_preprocessed.csv` (351 MB, 641,601 rows, 12.5 m grid).
  - If the file isn't present, fetch from `https://github.com/Somon8/mbml-forest-pipeline/releases/download/v1.2-data/out_10km_idx_preprocessed.csv` with curl.
  - Column reference: see `DATA_PIPELINE.md §5` and the spatial-features columns from §6.
- Environment: `uv` is configured at the repo root, Python ≥3.12. `pyro-ppl`, `torch`, `scikit-learn`, `pandas`, `numpy`, `scipy`, `matplotlib` are available. Add `joblib`, `psutil`, `tabulate` if missing. Do not pull in `gpytorch` or `lightgbm` unless absolutely required — keep the stack as Pyro + PyTorch + sklearn.

## Target variable

`y = volym_omdrev2 - volym_omdrev1`  (units: m³/ha)

Predict Δvolume, the change in stem volume between cycle 1 (2008–2016) and cycle 2 (2018–2025).

## Filtering

Apply the `is_stable_forest` mask before any modelling. This restricts the dataset to pixels that were forest in both cycles, are not lake, and did not lose volume/height/biomass between cycles. Definition (already materialised as a column in the preprocessed CSV; verify):

```
is_stable_forest = ~is_no_forest
                 & ~is_lake
                 & ~(delta_neg_medelhojd | delta_neg_p95
                     | delta_neg_medeldiameter | delta_neg_biomassa
                     | delta_neg_volym)
```

Expected size after filtering: roughly 300k–400k rows. Print the actual count and the per-BK distribution of surviving rows at the start.

## Features (leakage-aware)

Use **only features that are temporally consistent with omdrev1 or are time-invariant**. Any feature that is measured at or after the omdrev2 epoch is forbidden — including the spatial-features columns suffixed with `_omdrev2`, and `tradhojd` (which the journal notes correlates strongly with `medelhojd_omdrev2`, indicating it's contemporaneous).

**Default feature list (use these unless you have a reason not to):**

- `volym_omdrev1`, `biomassa_omdrev1`, `grundyta_omdrev1`, `medelhojd_omdrev1`, `medeldiameter_omdrev1`, `p95_omdrev1`, `vegetationskvot_omdrev1` — initial forest state
- `slu_skogskarta_gran_volym`, `slu_skogskarta_tall_volym`, `slu_skogskarta_bjork_volym` — species composition (drop ek, bok, contorta — too sparse / zero per the journal)
- `markfuktighet` (continuous) — soil moisture
- `log1p(flodesackumulering)` — flow accumulation, log-transformed per journal §2.7
- `x`, `y` — coordinates in EPSG:3006 metres (non-leaky, useful for GPs)

**Treat as an open question (decide and document in the notebook):** SLU Skogskarta layers may have been produced post-omdrev1. Species composition is biologically near-stationary on a decade scale, so this is probably acceptable, but note it explicitly.

**Forbidden features (would leak):** anything `*_omdrev2`, `tradhojd`, all spatial features that suffix `_omdrev2_mean3`, `_omdrev2_std3`, `_omdrev2_dNS`, `_omdrev2_dEW`, `_omdrev2_grad_*`. The distance features `dist_to_no_forest_m` and `dist_to_lake_m` are derived from `is_no_forest` (which uses both cycles) — exclude as a precaution.

Standardize all numeric features (mean 0, std 1) before training. Persist the scaler. **Important:** fit the scaler on the *training* fold of the largest scaling step only, and reuse for all smaller subsamples and the test set — keeps comparisons fair.

## Train / test split and scaling axis

The dataset has 441 BK indexruta cells (500 m × 500 m blocks; each contains up to 1,600 pixels at 12.5 m). Use BK as the spatial grouping unit.

1. **Test set (fixed):** randomly hold out 20% of BK cells (≈88 cells, ~140k pixels before filtering, fewer after). Seed: 42. This test set is reused at every scaling step and across every model.
2. **Train pool:** the remaining ~353 BK cells.
3. **Scaling axis:** train every model at five sizes by including a growing subset of train BK cells:
   - `n_cells ∈ {5, 25, 100, 250, all available}`
   - At each step, pick cells deterministically from a fixed shuffle (seed: 42) so that smaller subsets are nested inside larger ones. This makes the scaling curves monotonic in data, not in subsample variance.
4. **Exception for exact GP:** only run at `n_cells ∈ {5, 25}` (≤ ~10k training pixels). Skip larger sizes with a clear log message — exact GP at N=100k will OOM.
5. After all scaling runs at group-split, additionally train every model **once** on a random 80/20 pixel split at full N. Report this as a single extra row labelled `random_split` to quantify spatial leakage.

Save the split definition (BK cells in test, train cell ordering, random seeds) to `Projects/cache/splits.json` so reruns are deterministic.

## Models (seven total)

All Bayesian models implemented in **Pyro + PyTorch** with stochastic variational inference (SVI). Use `pyro.contrib.gp` for the GPs. No `gpytorch`. Frequentist baselines in `sklearn`.

1. **Linear regression (frequentist baseline)** — `sklearn.linear_model.LinearRegression`. No uncertainty. Sanity check + speed reference.
2. **Bayesian Linear Regression (BLR)** — Pyro model with Normal priors on weights, Normal likelihood, learned noise σ. SVI with mean-field guide.
3. **Hierarchical Bayesian regression** — Pyro model with partial pooling on `BK` (indexruta cell). Random intercept per cell with a hyperprior; fixed-slope features. SVI mean-field guide.
4. **Exact GP** — `pyro.contrib.gp.models.GPRegression` with Matérn-5/2 kernel, ARD lengthscales. Only runs at ≤25 BK cells (≤~10k pixels).
5. **Sparse Variational GP (SVGP)** — `pyro.contrib.gp.models.VariationalSparseGP` with ~512 inducing points (initialise via k-means on training inputs), Matérn-5/2 ARD kernel, Gaussian likelihood. Trained with Adam + minibatching.
6. **Bayesian Neural Network (BNN)** — Pyro model: 2 hidden layers, 64 units, ReLU, Normal priors on weights, learned σ on the output. Mean-field variational guide. Adam.
7. **Random Forest + Gradient Boosting (frequentist baselines)** — `sklearn.ensemble.RandomForestRegressor` and `sklearn.ensemble.HistGradientBoostingRegressor`. No epistemic uncertainty by default; for predictive intervals use the empirical std of leaf predictions for RF, and skip probabilistic metrics for GBM (or use quantile loss versions if trivial). Document either way.

(Yes, that's eight if you count linear regression as a separate model. Include it. The "seven" upstream is GP + BLR + SVGP + BNN + Hierarchical + RF + GBM; linear regression is a free baseline.)

### Common model API

Every model lives in `Projects/models/<name>.py` and inherits from `Projects/models/base.py:BaseModel`:

```python
class BaseModel(ABC):
    name: str
    is_probabilistic: bool

    def fit(self, X_train, y_train, *, group_train=None) -> dict:
        """Returns a dict with at least 'fit_seconds' and 'peak_memory_mb'."""

    def predict(self, X_test, *, n_samples=200) -> tuple[np.ndarray, np.ndarray | None]:
        """Returns (mean_predictions, std_predictions). std=None for non-probabilistic."""

    def predict_samples(self, X_test, n_samples=200) -> np.ndarray | None:
        """Returns (n_samples, n_test) draws from the posterior predictive, or None."""

    def save(self, path: Path): ...
    def load(self, path: Path): ...
```

`group_train` carries the BK code per training row for the hierarchical model only; other models ignore it.

For probabilistic metrics that need full predictive samples (CRPS, calibration), use `predict_samples` if available, otherwise fall back to assuming the predictive is Gaussian with the returned `(mean, std)`.

## Evaluation metrics

For every (model, n_cells) cell of the experiment matrix, compute:

**Point metrics:**
- RMSE, MAE, R² (sklearn)

**Probabilistic metrics** (skip for non-probabilistic models with a clear `n/a`):
- NLPD: mean negative log predictive density on the test set. For Gaussian predictive, `0.5 * log(2π σ²) + 0.5 * (y - μ)² / σ²`. Lower is better.
- CRPS: continuous ranked probability score. Use the closed-form formula for a Gaussian predictive with mean μ and std σ:
  ```
  CRPS = σ * (z * (2 Φ(z) − 1) + 2 φ(z) − 1/√π),  z = (y − μ) / σ
  ```
  Where Φ and φ are the standard Normal CDF and PDF. Lower is better.
- 90% prediction-interval coverage: fraction of test points where `y` falls inside `[μ − 1.645 σ, μ + 1.645 σ]`. Target: 0.90.
- Calibration plot: empirical coverage at α ∈ {0.1, 0.2, …, 0.95} vs nominal α. One figure per model at full N.

**Runtime / resource:**
- Wall-clock fit time (seconds), wall-clock predict time on full test set (seconds), peak resident memory during fit (MB, via `psutil.Process().memory_info().rss` sampled in a thread, or `tracemalloc` for a less invasive estimate). Document which.

**Residual diagnostics (full-N fits only):**
- Spatial residual map: scatter `(x, y)` colour-coded by `y_pred − y_true` for each model. Use a fixed diverging colour scale across models so they're directly comparable.
- Histogram of residuals + QQ-plot vs Normal.
- Residual vs predicted scatter (heteroscedasticity check).

## Outputs

Single notebook: `Projects/04_model_comparison.ipynb`. Sections:

1. **Setup & data loading** — read CSV, apply mask, print sizes, list features used.
2. **Train/test split construction** — show test BK cells overlaid on a map of the AOI, print sizes per scaling step.
3. **Feature inspection** — pairplot or correlation heatmap of features post-standardization (down-sample to 5k rows for plotting).
4. **Per-model training (one section per model)** — fit at smallest scaling step, show predictions vs truth, print metrics. Each section short; the meat is in the loop.
5. **Full experiment loop** — iterate over (model, n_cells), cache results to `Projects/cache/results/<model>_<n_cells>.pkl`. Skip already-cached entries on rerun. Aggregate into a single `results` DataFrame.
6. **Results table** — pivot table: rows = models, columns = (n_cells, metric). Render with `tabulate` for the report.
7. **Scaling figures** — three panels, log-x:
   - RMSE vs n_cells per model
   - NLPD vs n_cells per model (probabilistic only)
   - Fit time vs n_cells per model
8. **Calibration figures** — one calibration plot per probabilistic model at full N, on a single shared axis.
9. **Spatial residual maps** — grid of residual maps (one per model) at full N.
10. **Discussion section** — short markdown cells calling out the headline findings: which model wins on point accuracy, which on calibration, which on speed; how performance scales with N; whether the random-split row is dramatically better than the group-split equivalent (and what that says about spatial autocorrelation).

Cache *everything* expensive to `Projects/cache/`:
- `splits.json` — split definitions
- `data.parquet` — filtered, feature-extracted, standardized data
- `results/<model>_<n_cells>.pkl` — fit metadata + predictions + metrics
- `models/<model>_<n_cells>.pt` (or .joblib) — trained model artifacts

The notebook should re-run end to end in <1 minute on a warm cache.

## Acceptance checks

Before declaring done, verify:

- [ ] `is_stable_forest` filter applied; print pre/post row counts.
- [ ] No forbidden features in the feature matrix (`tradhojd`, anything `*_omdrev2`, distance-to-mask).
- [ ] Test BK cells are disjoint from every training subsample.
- [ ] Smaller scaling subsets are nested inside larger ones.
- [ ] Same `(scaler, test set)` used across all models.
- [ ] Exact GP skipped at large N with a logged reason, not silently failing.
- [ ] `random_split` row exists and shows substantially better metrics than `group_split` at the same N — if it doesn't, something is wrong with the split logic.
- [ ] Calibration: 90% intervals on the test set should be in the 0.85–0.95 range for well-specified models. If a Bayesian model is at 0.99 or 0.5, document why.
- [ ] All cached files reproducible from seeded RNG.
- [ ] Notebook runs top-to-bottom without errors on a clean kernel after running the experiment loop once.

## Anti-goals

- Do not invent new spatial features.
- Do not change the target definition.
- Do not introduce features not on the allowed list without explicitly flagging the leakage analysis.
- Do not use full-batch optimization for SVGP / BNN at large N — use minibatching with batch_size=4096.
- Do not let the notebook silently swallow OOM errors. If exact GP can't run, log it loudly.

## Style

- Prose markdown cells before each major code section explaining what the code does and why, in 2–4 sentences. The notebook is read by a course examiner.
- Plot titles and axis labels in English, units shown.
- Random seeds (42 throughout) at the top of every model file.

===

## Notes on running this prompt

- Run from a fresh `claude` session inside `Projects/`. Hand it the prompt verbatim.
- It will likely take an hour or two of supervised back-and-forth, mostly during the model-implementation sections (BNN priors, SVGP convergence, hierarchical model factor structure).
- If the first SVGP run looks underfit, the lever is `num_inducing` (raise to 1024) and `lr` (drop to 1e-3 with longer training). Keep Matérn-5/2 ARD as the default.
- If hierarchical model SVI doesn't converge, reduce learning rate and increase `num_iterations`; partial-pooling guides are sensitive.
- After the notebook is built, the report-quality figures live in `Projects/cache/figures/`. Re-export to PDF for the writeup.
