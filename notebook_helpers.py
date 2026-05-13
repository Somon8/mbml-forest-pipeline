"""Shared helpers for ``04_model_comparison.ipynb``.

Putting the heavy logic here keeps notebook cells short and means the
notebook can re-run end to end on a warm cache by orchestrating already
cached pieces.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS, Predictive
from pyro.infer.autoguide import AutoNormal, AutoDiagonalNormal
from pyro.optim import Adam
from scipy.stats import norm as _norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models import MODELS


SEED = 42
PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "cache"
CACHE_RESULTS = CACHE_DIR / "results"
CACHE_MODELS = CACHE_DIR / "models"
SPLITS_JSON = CACHE_DIR / "splits.json"
DATA_CACHE = CACHE_DIR / "data.pkl"
# Backwards-compatibility alias kept so older notebook cells still resolve.
DATA_PARQUET = DATA_CACHE

# Z-value for a central 90 % normal interval.
Z90 = 1.6448536269514722


# ----------------------------------------------------------------------
# Feature config
# ----------------------------------------------------------------------

# The feature list is leakage-aware: only omdrev1 / time-invariant /
# log-transformed flow + species-composition columns. See PROJECT_JOURNAL.md
# §2.7 (log1p flow) and the task brief for the full justification.
FEATURE_COLS = [
    "volym_omdrev1",
    "biomassa_omdrev1",
    "grundyta_omdrev1",
    "medelhojd_omdrev1",
    "medeldiameter_omdrev1",
    "p95_omdrev1",
    "vegetationskvot_omdrev1",
    "slu_skogskarta_gran_volym",
    "slu_skogskarta_tall_volym",
    "slu_skogskarta_bjork_volym",
    "markfuktighet",
    "log1p_flodesackumulering",
    "x",
    "y",
]

FORBIDDEN_FEATURES = {
    # any *_omdrev2 column
    "tradhojd",
    "dist_to_no_forest_m",
    "dist_to_lake_m",
}

TARGET_COL = "delta_volym"


# ----------------------------------------------------------------------
# Data loading and split construction
# ----------------------------------------------------------------------

def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def apply_filter_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ``is_stable_forest`` and add the engineered columns we need."""

    # Sanity: rebuild is_stable_forest from primitives, just to be defensive.
    primitives = [
        "is_no_forest", "is_lake",
        "delta_neg_medelhojd", "delta_neg_p95", "delta_neg_medeldiameter",
        "delta_neg_biomassa", "delta_neg_volym",
    ]
    for c in primitives:
        if c not in df.columns:
            raise ValueError(f"missing preprocessing column: {c}")
    rebuilt = (
        ~df["is_no_forest"].astype(bool)
        & ~df["is_lake"].astype(bool)
        & ~(df["delta_neg_medelhojd"].astype(bool)
            | df["delta_neg_p95"].astype(bool)
            | df["delta_neg_medeldiameter"].astype(bool)
            | df["delta_neg_biomassa"].astype(bool)
            | df["delta_neg_volym"].astype(bool))
    )
    if "is_stable_forest" in df.columns:
        cur = df["is_stable_forest"].astype(bool)
        if not (rebuilt == cur).all():
            df = df.copy()
            df["is_stable_forest"] = rebuilt
    else:
        df["is_stable_forest"] = rebuilt

    n0 = len(df)
    df = df.loc[df["is_stable_forest"]].copy()
    n1 = len(df)

    df[TARGET_COL] = df["volym_omdrev2"] - df["volym_omdrev1"]
    df["log1p_flodesackumulering"] = np.log1p(df["flodesackumulering"])

    print(f"Rows before mask: {n0:,}")
    print(f"Rows after  mask: {n1:,}")

    bk_counts = df["BK"].value_counts()
    print(f"BK cells with ≥1 surviving pixel: {len(bk_counts):,}")
    print(f"  median pixels per BK: {bk_counts.median():.0f}")
    print(f"  min/max pixels per BK: {bk_counts.min()} / {bk_counts.max()}")

    return df


def assert_no_forbidden(feature_cols):
    bad = []
    for c in feature_cols:
        if c in FORBIDDEN_FEATURES:
            bad.append(c)
        if c.endswith("_omdrev2") or "_omdrev2_" in c:
            bad.append(c)
    if bad:
        raise AssertionError(f"forbidden leaky features: {bad}")


@dataclass
class Splits:
    test_bk: list[str]
    train_bk_ordered: list[str]   # nested order (deterministic shuffle)
    n_cells_grid: list[int | str]
    seed: int = SEED

    def to_dict(self):
        return {
            "test_bk": list(self.test_bk),
            "train_bk_ordered": list(self.train_bk_ordered),
            "n_cells_grid": list(self.n_cells_grid),
            "seed": self.seed,
        }


def build_splits(df: pd.DataFrame, n_cells_grid=(5, 25, 100, 250, "all"),
                 test_frac: float = 0.20, seed: int = SEED) -> Splits:
    """Hold out 20 % of BK cells as test, deterministically order the rest
    so that smaller scaling subsets are nested inside larger ones."""
    bk_all = sorted(df["BK"].unique().tolist())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(bk_all))
    n_test = int(round(test_frac * len(bk_all)))
    test_bk = sorted([bk_all[i] for i in perm[:n_test]])
    train_pool = [bk_all[i] for i in perm[n_test:]]
    # Second deterministic shuffle of the remaining training pool — fixed
    # so that {first 5, first 25, ...} are nested.
    rng2 = np.random.default_rng(seed + 1)
    train_perm = rng2.permutation(len(train_pool))
    train_bk_ordered = [train_pool[i] for i in train_perm]
    return Splits(
        test_bk=test_bk,
        train_bk_ordered=train_bk_ordered,
        n_cells_grid=list(n_cells_grid),
        seed=seed,
    )


def save_splits(splits: Splits, path: Path = SPLITS_JSON):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits.to_dict(), indent=2))


def load_splits(path: Path = SPLITS_JSON) -> Splits:
    d = json.loads(path.read_text())
    return Splits(**d)


def get_train_bk(splits: Splits, n_cells) -> list[str]:
    if n_cells == "all":
        return list(splits.train_bk_ordered)
    return list(splits.train_bk_ordered[: int(n_cells)])


# ----------------------------------------------------------------------
# Feature matrix construction
# ----------------------------------------------------------------------

def split_frames(df: pd.DataFrame, splits: Splits, n_cells):
    train_bk = set(get_train_bk(splits, n_cells))
    test_bk = set(splits.test_bk)
    df_train = df[df["BK"].isin(train_bk)].copy()
    df_test = df[df["BK"].isin(test_bk)].copy()
    return df_train, df_test


def matrices(df_part: pd.DataFrame, feature_cols=FEATURE_COLS):
    X = df_part[feature_cols].to_numpy(dtype=np.float32)
    y = df_part[TARGET_COL].to_numpy(dtype=np.float32)
    g = df_part["BK"].to_numpy()
    coords = df_part[["x", "y"]].to_numpy(dtype=np.float64)
    return X, y, g, coords


def fit_scaler(X_train: np.ndarray):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return mu.astype(np.float32), sigma.astype(np.float32)


def apply_scaler(X, mu, sigma):
    return ((X - mu) / sigma).astype(np.float32)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def gaussian_nlpd(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2)
                         + 0.5 * (y - mu) ** 2 / sigma ** 2))


def gaussian_crps(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    pdf = _norm.pdf(z)
    cdf = _norm.cdf(z)
    crps = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def coverage(y, mu, sigma, alpha=0.90):
    z = _norm.ppf(0.5 + alpha / 2.0)
    sigma = np.maximum(sigma, 1e-6)
    inside = (y >= mu - z * sigma) & (y <= mu + z * sigma)
    return float(np.mean(inside))


def calibration_curve(y, mu, sigma, alphas=None):
    if alphas is None:
        alphas = np.arange(0.1, 0.96, 0.05)
    sigma = np.maximum(sigma, 1e-6)
    emp = []
    for a in alphas:
        z = _norm.ppf(0.5 + a / 2.0)
        inside = (y >= mu - z * sigma) & (y <= mu + z * sigma)
        emp.append(float(np.mean(inside)))
    return np.asarray(alphas), np.asarray(emp)


def point_metrics(y, mu):
    rmse = float(np.sqrt(mean_squared_error(y, mu)))
    mae = float(mean_absolute_error(y, mu))
    r2 = float(r2_score(y, mu))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ----------------------------------------------------------------------
# Experiment loop primitive
# ----------------------------------------------------------------------

def cache_path(model_name: str, label: str) -> Path:
    CACHE_RESULTS.mkdir(parents=True, exist_ok=True)
    return CACHE_RESULTS / f"{model_name}__{label}.pkl"


def model_blob_path(model_name: str, label: str) -> Path:
    CACHE_MODELS.mkdir(parents=True, exist_ok=True)
    # Pyro models save as .pt, sklearn as .joblib; both via .save() so the
    # caller picks. We just use .bin as a generic suffix.
    return CACHE_MODELS / f"{model_name}__{label}.bin"


def run_one(
    model_name: str,
    label: str,
    X_train, y_train, g_train,
    X_test, y_test, g_test, coords_test,
    *,
    is_random_split: bool = False,
    n_cells_eff: int | str = None,
    n_train_bk: int | None = None,
    force: bool = False,
    n_pred_samples: int = 200,
    model_kwargs: dict | None = None,
):
    """Fit one model on one training subset, score on the test set, cache.

    The cache key is ``<model>__<label>``. Re-run with ``force=True`` to
    overwrite. Returns the cached dict (predictions + metrics + meta).
    """
    out_path = cache_path(model_name, label)
    if out_path.exists() and not force:
        with out_path.open("rb") as f:
            return pickle.load(f)

    print(f"  fitting {model_name}@{label} (n_train={len(X_train):,}) ...",
          flush=True)
    cls = MODELS[model_name]
    model = cls(**(model_kwargs or {}))
    fit_info = model.fit(
        X_train, y_train,
        group_train=g_train if model_name == "hierarchical" else None,
    )

    t_pred = time.perf_counter()
    if model_name == "hierarchical":
        mu, sigma = model.predict(X_test, n_samples=n_pred_samples,
                                  group_test=g_test)
    else:
        mu, sigma = model.predict(X_test, n_samples=n_pred_samples)
    pred_seconds = time.perf_counter() - t_pred

    metrics = point_metrics(y_test, mu)
    if sigma is not None:
        metrics["nlpd"] = gaussian_nlpd(y_test, mu, sigma)
        metrics["crps"] = gaussian_crps(y_test, mu, sigma)
        metrics["coverage_90"] = coverage(y_test, mu, sigma, alpha=0.90)
        alphas, emp = calibration_curve(y_test, mu, sigma)
    else:
        metrics["nlpd"] = np.nan
        metrics["crps"] = np.nan
        metrics["coverage_90"] = np.nan
        alphas, emp = None, None

    blob_path = model_blob_path(model_name, label)
    try:
        model.save(blob_path)
    except Exception as e:
        print(f"    note: {model_name} save failed ({e}); skipping artifact")
        blob_path = None

    record = {
        "model": model_name,
        "label": label,
        "is_random_split": bool(is_random_split),
        "n_cells_eff": n_cells_eff,
        "n_train_bk": n_train_bk,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "fit_seconds": float(fit_info.get("fit_seconds", np.nan)),
        "predict_seconds": float(pred_seconds),
        "peak_memory_mb": float(fit_info.get("peak_memory_mb", np.nan)),
        "delta_memory_mb": float(fit_info.get("delta_memory_mb", np.nan)),
        "metrics": metrics,
        "y_test": np.asarray(y_test, dtype=np.float32),
        "mu": np.asarray(mu, dtype=np.float32),
        "sigma": (None if sigma is None else np.asarray(sigma, dtype=np.float32)),
        "coords_test": np.asarray(coords_test, dtype=np.float64),
        "alphas": alphas,
        "emp_coverage": emp,
        "model_artifact": str(blob_path) if blob_path else None,
    }
    with out_path.open("wb") as f:
        pickle.dump(record, f)
    print(f"    rmse={metrics['rmse']:6.2f}  fit={record['fit_seconds']:.1f}s"
          f"  peak_rss={record['peak_memory_mb']:.0f}MB", flush=True)
    return record


def aggregate_results():
    """Concat all cached result pickles into one DataFrame (no predictions)."""
    rows = []
    for p in sorted(CACHE_RESULTS.glob("*.pkl")):
        with p.open("rb") as f:
            rec = pickle.load(f)
        flat = {
            "model": rec["model"],
            "label": rec["label"],
            "is_random_split": rec["is_random_split"],
            "n_cells_eff": rec["n_cells_eff"],
            "n_train_bk": rec["n_train_bk"],
            "n_train": rec["n_train"],
            "n_test": rec["n_test"],
            "fit_seconds": rec["fit_seconds"],
            "predict_seconds": rec["predict_seconds"],
            "peak_memory_mb": rec["peak_memory_mb"],
            "delta_memory_mb": rec["delta_memory_mb"],
        }
        for k, v in rec["metrics"].items():
            flat[k] = v
        rows.append(flat)
    return pd.DataFrame(rows)


def load_record(model_name: str, label: str) -> dict:
    p = cache_path(model_name, label)
    with p.open("rb") as f:
        return pickle.load(f)



# ======================================================================
# Factor analysis (used by final_notebook.ipynb §4)
# ======================================================================

# Six collinear lidar views. Anchor order matters for identification:
#   row 0 (biomass)      pins z_mass at lambda = 1, lambda_geom = 0
#   row 1 (mean_height)  anchors z_geom with a HalfNormal(1) positive loading
# Anderson–Rubin / Geweke–Zhou constraint: Lambda lower-triangular in first J rows.
FACTOR_VIEWS = ['biomass_t1', 'mean_height_t1',
                'basal_area_t1', 'volume_t1',
                'mean_diameter_t1', 'p95_height_t1']
VIEW_LABELS  = ['bio', 'CH', 'BA', 'vol', 'diam', 'p95']
K_DEFAULT, J_DEFAULT = 6, 2


def make_factor_model(K: int = K_DEFAULT, J: int = J_DEFAULT):
    """Return a Pyro factor model closure parameterised by (K, J)."""
    def factor_model(Y_obs=None, N=None):
        if Y_obs is not None:
            N = Y_obs.shape[0]
        lam_10   = pyro.sample('lam_10',   dist.Normal(0., 1.))                                  # CH on z_mass (free)
        lam_11   = pyro.sample('lam_11',   dist.HalfNormal(1.))                                  # CH on z_geom (positive anchor)
        lam_rest = pyro.sample('lam_rest', dist.Normal(torch.zeros(K - 2, J), 1.).to_event(2))
        ones  = torch.ones_like(lam_10).unsqueeze(-1)
        zeros = torch.zeros_like(lam_10).unsqueeze(-1)
        row0  = torch.cat([ones, zeros], dim=-1)
        row1  = torch.stack([lam_10, lam_11], dim=-1)
        Lam   = torch.cat([row0.unsqueeze(-2), row1.unsqueeze(-2), lam_rest], dim=-2)
        mu    = pyro.sample('mu',    dist.Normal(torch.zeros(K), 1.).to_event(1))
        sigma = pyro.sample('sigma', dist.HalfNormal(torch.ones(K)).to_event(1))
        with pyro.plate('rows', N):
            z   = pyro.sample('z', dist.Normal(torch.zeros(J), 1.).to_event(1))
            loc = mu + (z.unsqueeze(-2) * Lam.unsqueeze(-3)).sum(-1)
            pyro.sample('y', dist.Normal(loc, sigma).to_event(1), obs=Y_obs)
    return factor_model


def fit_factor_model(Y_t: torch.Tensor, K: int = K_DEFAULT, J: int = J_DEFAULT,
                     n_steps: int = 6000, lr: float = 5e-3, seed: int = 42,
                     verbose: bool = True):
    """Fit the 2-factor Bayesian FA via SVI on standardised training views.

    Returns
    -------
    guide : AutoNormal
    losses : list[float]
    factor_model : callable  (the model closure, needed downstream for Predictive)
    """
    factor_model = make_factor_model(K=K, J=J)
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    guide = AutoNormal(factor_model)
    svi   = SVI(factor_model, guide, Adam({'lr': lr}), loss=Trace_ELBO())
    losses = []
    for step in range(n_steps):
        losses.append(svi.step(Y_obs=Y_t))
        if verbose and step % 1000 == 0:
            print(f'  step {step:>5d}  -ELBO = {losses[-1]:,.0f}')
    if verbose:
        print(f'  step {len(losses) - 1:>5d}  -ELBO = {losses[-1]:,.0f}')
    return guide, losses, factor_model


def _strip_particle_dim(t: torch.Tensor) -> torch.Tensor:
    # AutoNormal samples insert a singleton particle dim at position 1; strip it.
    return t.squeeze(1) if t.dim() >= 2 else t


def posterior_loadings(factor_model, guide, Y_t: torch.Tensor,
                       K: int = K_DEFAULT, J: int = J_DEFAULT,
                       view_labels: list[str] | None = None,
                       num_samples: int = 400) -> dict:
    """Sample the posterior and assemble the full (K, J) loading matrix.

    Returns a dict with: Lam_post, sig_post, mu_post (per-sample),
    Lam_mean, Lam_sd, sig_mean, mu_mean (posterior means / sds),
    and loading_df (pandas DataFrame for printing).
    """
    post = Predictive(
        factor_model, guide=guide, num_samples=num_samples,
        return_sites=['lam_10', 'lam_11', 'lam_rest', 'mu', 'sigma'],
    )(Y_obs=Y_t)

    lam10   = _strip_particle_dim(post['lam_10']).numpy()
    lam11   = _strip_particle_dim(post['lam_11']).numpy()
    lamrest = _strip_particle_dim(post['lam_rest']).numpy()
    mu_post = _strip_particle_dim(post['mu']).numpy()
    sig_pos = _strip_particle_dim(post['sigma']).numpy()

    S = lam10.shape[0]
    Lam_post = np.zeros((S, K, J), dtype=np.float32)
    Lam_post[:, 0, 0] = 1.0
    Lam_post[:, 1, 0] = lam10
    Lam_post[:, 1, 1] = lam11
    Lam_post[:, 2:, :] = lamrest

    Lam_mean = Lam_post.mean(0); Lam_sd = Lam_post.std(0)
    sig_mean = sig_pos.mean(0);  mu_mean = mu_post.mean(0)

    labels = view_labels if view_labels is not None else VIEW_LABELS
    loading_df = pd.DataFrame({
        'view':         labels,
        'lam_mass':     Lam_mean[:, 0].round(3),
        'lam_mass_sd':  Lam_sd[:,   0].round(3),
        'lam_geom':     Lam_mean[:, 1].round(3),
        'lam_geom_sd':  Lam_sd[:,   1].round(3),
        'sigma':        sig_mean.round(3),
        'frac_var':    ((Lam_mean[:, 0]**2 + Lam_mean[:, 1]**2) /
                        (Lam_mean[:, 0]**2 + Lam_mean[:, 1]**2 + sig_mean**2)).round(3),
    })
    return dict(Lam_post=Lam_post, sig_post=sig_pos, mu_post=mu_post,
                Lam_mean=Lam_mean, Lam_sd=Lam_sd,
                sig_mean=sig_mean, mu_mean=mu_mean,
                loading_df=loading_df)


def implied_corr(Lam: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Posterior-mean correlation matrix implied by (Lambda, sigma).

    Cov = Lambda Lambda^T + diag(sigma^2). Lam: (S, K, J), sigma: (S, K).
    Returns: (K, K) posterior-mean correlation matrix.
    """
    Cov  = (np.einsum('skj,slj->skl', Lam, Lam)
            + sigma[:, :, None]**2 * np.eye(Lam.shape[1])[None, :, :])
    diag = np.diagonal(Cov, axis1=-2, axis2=-1)
    norm = np.sqrt(diag[:, :, None] * diag[:, None, :])
    return (Cov / norm).mean(0)


def score_factors_ar(Y_raw: np.ndarray,
                     Y_mean_views: np.ndarray, Y_std_views: np.ndarray,
                     Lam_mean: np.ndarray, sig_mean: np.ndarray,
                     mu_mean: np.ndarray, J: int = J_DEFAULT) -> np.ndarray:
    """Closed-form Anderson–Rubin factor scoring for every row.

    z_hat[n] = (I + Lambda^T D^-1 Lambda)^-1 Lambda^T D^-1 (y_n - mu),
    D = diag(sigma^2). Standardisation uses train statistics passed in.
    """
    Y_z     = (Y_raw - Y_mean_views) / Y_std_views
    D_inv   = np.diag(1.0 / (sig_mean ** 2))
    prec_z  = np.eye(J) + Lam_mean.T @ D_inv @ Lam_mean
    M_score = np.linalg.solve(prec_z, Lam_mean.T @ D_inv)
    return (Y_z - mu_mean) @ M_score.T


# ======================================================================
# Bayesian regression (used by final_notebook.ipynb §8)
# ======================================================================

# ---------------------------------------------------------------------
# Pyro models
# ---------------------------------------------------------------------

def blr_pyro_model(X, y=None):
    """Bayesian linear regression with weak N(0,1) priors on standardised features.

        alpha       ~ N(0, 1)
        beta_k      ~ N(0, 1)   for k = 1 .. d
        sigma       ~ HalfNormal(1)
        y_n | X_n   ~ N(alpha + X_n . beta, sigma^2)
    """
    n, d = X.shape
    alpha = pyro.sample("alpha", dist.Normal(torch.tensor(0.0), torch.tensor(1.0)))
    beta  = pyro.sample("beta",  dist.Normal(torch.zeros(d), torch.ones(d)).to_event(1))
    sigma = pyro.sample("sigma", dist.HalfNormal(torch.tensor(1.0)))
    mean = alpha + X @ beta
    with pyro.plate("data", n):
        return pyro.sample("y", dist.Normal(mean, sigma), obs=y)


def hetero_pyro_model(X, y=None):
    """Heteroscedastic BLR: mean and log-variance each linear in X.

        alpha_mu, alpha_v ~ N(0, 1)
        beta_mu_k, beta_v_k ~ N(0, 1)
        log sigma^2(X_n)    = alpha_v + X_n . beta_v
        y_n | X_n           ~ N(alpha_mu + X_n . beta_mu, sigma^2(X_n))
    """
    n, d = X.shape
    alpha_mu = pyro.sample("alpha_mu", dist.Normal(torch.tensor(0.0), torch.tensor(1.0)))
    beta_mu  = pyro.sample("beta_mu",  dist.Normal(torch.zeros(d), torch.ones(d)).to_event(1))
    alpha_v  = pyro.sample("alpha_v",  dist.Normal(torch.tensor(0.0), torch.tensor(1.0)))
    beta_v   = pyro.sample("beta_v",   dist.Normal(torch.zeros(d), torch.ones(d)).to_event(1))
    mean    = alpha_mu + X @ beta_mu
    log_var = alpha_v + X @ beta_v
    sigma   = torch.exp(0.5 * log_var)
    with pyro.plate("data", n):
        return pyro.sample("y", dist.Normal(mean, sigma), obs=y)


# ---------------------------------------------------------------------
# Closed-form conjugate BLR
# ---------------------------------------------------------------------

def exact_blr_fit(s: dict, n_samples: int = 1000, tau: float = 1.0,
                  a0: float = 1e-2, b0: float = 1e-2, seed: int = 42):
    """Normal-Inverse-Gamma conjugate posterior for BLR on one slice_step payload.

    Returns
    -------
    theta_samples : (n_samples, d+1)  posterior draws of [alpha, beta]
    sigma_samples : (n_samples,)      posterior draws of sigma
    y_hat         : (n_test,)         posterior-mean prediction in original units
    y_true        : (n_test,)         test target in original units
    wall_time_s   : float             fit time (excludes prediction).
    """
    rng = np.random.default_rng(seed)
    X_tr = np.asarray(s["X_train"]); y_tr = np.asarray(s["y_train"]).ravel()
    X_te = np.asarray(s["X_test"]);  y_te = np.asarray(s["y_test"]).ravel()
    n, d = X_tr.shape
    X_tr_d = np.hstack([np.ones((n, 1)), X_tr])
    X_te_d = np.hstack([np.ones((X_te.shape[0], 1)), X_te])
    p = d + 1

    t0 = time.perf_counter()
    V0_inv = np.eye(p) / (tau ** 2)
    XtX = X_tr_d.T @ X_tr_d
    Xty = X_tr_d.T @ y_tr
    Vn_inv = XtX + V0_inv
    Vn = np.linalg.inv(Vn_inv)
    mu_n = Vn @ Xty
    a_n = a0 + n / 2.0
    b_n = b0 + 0.5 * (y_tr @ y_tr - mu_n @ Vn_inv @ mu_n)
    sigma2_samples = 1.0 / rng.gamma(shape=a_n, scale=1.0 / b_n, size=n_samples)
    L = np.linalg.cholesky(Vn)
    z = rng.normal(size=(n_samples, p))
    theta_samples = mu_n[None, :] + np.sqrt(sigma2_samples)[:, None] * (z @ L.T)
    wall = time.perf_counter() - t0

    y_hat_std = (X_te_d @ theta_samples.T).mean(axis=1)
    y_hat  = y_hat_std * s["y_std"] + s["y_mean"]
    y_true = y_te      * s["y_std"] + s["y_mean"]
    return theta_samples, np.sqrt(sigma2_samples), y_hat, y_true, wall


# ---------------------------------------------------------------------
# SVI (AutoDiagonalNormal guide) — works on any Pyro model with the
# standard ``model(X, y=None)`` signature
# ---------------------------------------------------------------------

def svi_fit(s: dict, model_fn, n_steps: int = 2000, n_posterior: int = 1000,
            lr: float = 0.01,
            return_sites=("alpha", "beta", "sigma"),
            mean_site: str = "alpha", weight_site: str = "beta",
            seed: int = 42):
    """Fit any Pyro model with mean-field SVI on a slice_step payload.

    Returns
    -------
    posterior     : dict[str, np.ndarray]   one entry per ``return_sites``
    elbo_trace    : list[float]
    y_hat         : (n_test,)               posterior-mean prediction in original units
    y_true        : (n_test,)
    wall_time_s   : float                   fit time (excludes Predictive sampling).
    """
    X_tr_t = torch.tensor(s["X_train"], dtype=torch.float32)
    y_tr_t = torch.tensor(s["y_train"], dtype=torch.float32)

    pyro.clear_param_store()
    pyro.set_rng_seed(seed); torch.manual_seed(seed)
    guide = AutoDiagonalNormal(model_fn)
    svi   = SVI(model_fn, guide, Adam({"lr": lr}), loss=Trace_ELBO())

    elbo = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        elbo.append(svi.step(X_tr_t, y_tr_t))
    wall = time.perf_counter() - t0

    predictive = Predictive(model_fn, guide=guide, num_samples=n_posterior,
                            return_sites=return_sites)
    posterior = {k: v.detach().numpy() for k, v in predictive(X_tr_t, y_tr_t).items()}

    alpha_post = posterior[mean_site].reshape(-1)
    beta_post  = posterior[weight_site].reshape(n_posterior, -1)
    y_hat_std  = alpha_post.mean() + s["X_test"] @ beta_post.mean(axis=0)
    y_hat  = y_hat_std * s["y_std"] + s["y_mean"]
    y_true = s["y_test"] * s["y_std"] + s["y_mean"]
    return posterior, elbo, y_hat, y_true, wall


# ---------------------------------------------------------------------
# NUTS — assumes homoscedastic BLR sites (alpha, beta, sigma)
# ---------------------------------------------------------------------

def nuts_fit(s: dict, model_fn, warmup: int = 500, draws: int = 500,
             seed: int = 42):
    """Fit a Pyro model with NUTS on a slice_step payload.

    Returns posterior dict (numpy arrays), y_hat, y_true, wall_time_s.
    """
    X_tr_t = torch.tensor(s["X_train"], dtype=torch.float32)
    y_tr_t = torch.tensor(s["y_train"], dtype=torch.float32)

    pyro.clear_param_store()
    pyro.set_rng_seed(seed); torch.manual_seed(seed)
    kernel = NUTS(model_fn)
    mcmc = MCMC(kernel, num_samples=draws, warmup_steps=warmup, num_chains=1)

    t0 = time.perf_counter()
    mcmc.run(X_tr_t, y_tr_t)
    wall = time.perf_counter() - t0

    posterior = {k: v.detach().numpy() for k, v in mcmc.get_samples().items()}
    alpha_post = posterior["alpha"]
    beta_post  = posterior["beta"]
    y_hat_std  = alpha_post.mean() + s["X_test"] @ beta_post.mean(axis=0)
    y_hat  = y_hat_std * s["y_std"] + s["y_mean"]
    y_true = s["y_test"] * s["y_std"] + s["y_mean"]
    return posterior, y_hat, y_true, wall


# ======================================================================
# Comparison-figure helpers (used by final_notebook.ipynb §9)
# ======================================================================

def _extract_payload_value(payload, key):
    """`wall_time_s` is stored at the top level by register_run; metrics are nested."""
    return payload["metrics"].get(key, payload.get(key))


def scaling_axis_figure(models, panels=None):
    """4-panel scaling-axis figure: RMSE / R² / Moran's I / wall time vs n_train."""
    import matplotlib.pyplot as plt
    if panels is None:
        panels = [("RMSE",         "RMSE (m³/ha)",         False),
                  ("R2",           "R²",                   False),
                  ("MoranI",       "Moran's I (residuals)", False),
                  ("wall_time_s",  "wall time (s)",         True)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4),
                              constrained_layout=True)
    for ax, (metric, label, log_y) in zip(axes, panels):
        for model_name, by_n in models.items():
            items = sorted(by_n.items(), key=lambda kv: by_n[kv[0]]["n_train"])
            xs, ys = [], []
            for _, v in items:
                val = _extract_payload_value(v, metric)
                if val is None:
                    continue
                xs.append(v["n_train"]); ys.append(val)
            if xs:
                ax.plot(xs, ys, marker="o", label=model_name, alpha=0.85)
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("training pixels")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3, which="both")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Test-set metrics + compute vs training-set size", fontsize=11)
    plt.show()
    return fig


def forest_plot(models, feature_names, target_step="all"):
    """Posterior coefficient forest plot (90% CIs). Skips models whose feature
    schema doesn't match feature_names (e.g. OLS-factor has a different design)."""
    import matplotlib.pyplot as plt
    feature_labels_full = ["intercept"] + list(feature_names)
    n_expected = len(feature_labels_full)

    eligible = [m for m in models
                if target_step in models[m]
                and models[m][target_step]["coefs_samples"] is not None
                and models[m][target_step]["coefs_samples"].shape[-1] == n_expected]
    skipped = [m for m in models
               if target_step in models[m]
               and models[m][target_step]["coefs_samples"] is not None
               and models[m][target_step]["coefs_samples"].shape[-1] != n_expected]
    if skipped:
        print(f"forest plot: skipping models with different feature schema -> {skipped}")

    n_models = len(eligible)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.4 * n_expected)),
                            constrained_layout=True)
    for i in range(0, n_expected, 2):
        ax.axhspan(i - 0.5, i + 0.5, color="gray", alpha=0.07, zorder=0)
    for i in range(n_expected - 1):
        ax.axhline(i + 0.5, color="gray", alpha=0.25, linewidth=0.5, zorder=0)

    offsets = np.linspace(-0.3, 0.3, n_models) if n_models > 1 else [0.0]
    colors  = plt.cm.tab10(np.linspace(0, 1, max(n_models, 2)))

    for idx, name in enumerate(eligible):
        coefs = models[name][target_step]["coefs_samples"]
        means = coefs.mean(axis=0)
        if coefs.shape[0] > 1:
            lo = np.quantile(coefs, 0.05, axis=0)
            hi = np.quantile(coefs, 0.95, axis=0)
        else:
            lo = hi = means
        y_pos = np.arange(n_expected) + offsets[idx]
        ax.errorbar(means, y_pos, xerr=[means - lo, hi - means],
                     fmt="o", capsize=3, color=colors[idx],
                     label=name, alpha=0.85, zorder=2)
    ax.axvline(0, color="k", linestyle=":", alpha=0.5, zorder=1)
    ax.set_yticks(np.arange(n_expected))
    ax.set_yticklabels(feature_labels_full)
    ax.set_ylim(n_expected - 0.5, -0.5)
    ax.set_xlabel("standardised coefficient (intercept + β)")
    ax.set_title(f"Posterior 90% CIs at n_cells = {target_step!r} — OLS shown without uncertainty")
    ax.legend(loc="best", fontsize=8)
    plt.show()
    return fig


def predicted_vs_actual_figure(models, slice_step_fn, target_step="all"):
    """Scatter grid of predicted vs actual y on the test set, one panel per model."""
    import matplotlib.pyplot as plt
    s = slice_step_fn(target_step)
    y_true = s["y_test"] * s["y_std"] + s["y_mean"]
    eligible = [m for m in models
                if target_step in models[m]
                and models[m][target_step].get("y_pred_test") is not None]
    n_cols = 3
    n_rows = (len(eligible) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                              constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    xmin = float(min(y_true.min(),
                      min(models[m][target_step]["y_pred_test"].min() for m in eligible)))
    xmax = float(max(y_true.max(),
                      max(models[m][target_step]["y_pred_test"].max() for m in eligible)))
    for i, name in enumerate(eligible):
        ax = axes[i]
        y_pred = models[name][target_step]["y_pred_test"]
        m = models[name][target_step]["metrics"]
        ax.scatter(y_true, y_pred, s=2, alpha=0.15, rasterized=True)
        ax.plot([xmin, xmax], [xmin, xmax], "k--", alpha=0.5, linewidth=1)
        ax.set_title(f"{name}  (RMSE = {m['RMSE']:.1f}, R² = {m['R2']:.3f})")
        if i % n_cols == 0:
            ax.set_ylabel("predicted ΔV (m³/ha)")
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("actual ΔV (m³/ha)")
    for ax in axes[len(eligible):]:
        ax.set_visible(False)
    fig.suptitle(f"Predicted vs. actual (n_cells = {target_step!r})", fontsize=11)
    plt.show()
    return fig


def posterior_predictive_coverage(models, slice_step_fn, name, n_cells,
                                  alpha_levels, seed=42):
    """Empirical coverage at every nominal credible level for one (model, step).

    Two paths: linear models with (coefs_samples, sigma_samples) reconstruct
    posterior predictives analytically; non-linear models (BNN, cluster) skip
    coefs and pass ``y_pred_samples_test`` in original units directly.
    """
    s = slice_step_fn(n_cells)
    payload = models[name][n_cells]
    coefs = payload.get("coefs_samples")
    if coefs is not None and coefs.shape[0] >= 2 and coefs.shape[-1] == 1 + s["X_test"].shape[1]:
        sigma = payload.get("sigma_samples")
        alpha_samples = coefs[:, 0]
        beta_samples  = coefs[:, 1:]
        mean_std = alpha_samples[:, None] + beta_samples @ s["X_test"].T
        if name == "Heteroscedastic":
            alpha_v = payload["alpha_v_samples"]
            beta_v  = payload["beta_v_samples"]
            log_var  = alpha_v[:, None] + beta_v @ s["X_test"].T
            sigma_std = np.exp(0.5 * log_var)
        elif sigma is not None:
            sigma_std = sigma[:, None]
        else:
            return None
        rng = np.random.default_rng(seed)
        pp_std = mean_std + sigma_std * rng.normal(size=mean_std.shape)
        pp = pp_std * s["y_std"] + s["y_mean"]
    else:
        pp = payload.get("y_pred_samples_test")
        if pp is None:
            return None
    y_true = s["y_test"] * s["y_std"] + s["y_mean"]
    coverage_curve = []
    for a in alpha_levels:
        lo = np.quantile(pp, (1 - a) / 2, axis=0)
        hi = np.quantile(pp, (1 + a) / 2, axis=0)
        coverage_curve.append(float(((y_true >= lo) & (y_true <= hi)).mean()))
    return np.array(coverage_curve)


def calibration_figure(models, slice_step_fn,
                       candidate_models=("Exact", "SVI", "MCMC", "Heteroscedastic"),
                       alpha_levels=None, seed=42):
    """Empirical vs nominal posterior-predictive coverage, one line per Bayesian model."""
    import matplotlib.pyplot as plt
    if alpha_levels is None:
        alpha_levels = np.arange(0.1, 1.0, 0.1)
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfectly calibrated")
    ax.grid(True, alpha=0.3)
    for name in candidate_models:
        if name not in models:
            continue
        n_pick = max(models[name], key=lambda k: models[name][k]["n_train"])
        cov = posterior_predictive_coverage(models, slice_step_fn, name, n_pick,
                                            alpha_levels, seed=seed)
        if cov is None:
            continue
        ax.plot(alpha_levels, cov, marker="o", label=f"{name}  (n_cells = {n_pick!r})", alpha=0.85)
    ax.set_xlabel("nominal credible level α")
    ax.set_ylabel("empirical coverage on test set")
    ax.set_title("Posterior-predictive calibration\n"
                  "(point models OLS / GBM cannot appear — no predictive distribution)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.show()
    return fig


def spatial_residual_maps(models, slice_step_fn, target_step="all"):
    """One residual-map panel per model; shared diverging colour scale."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    s = slice_step_fn(target_step)
    y_true = s["y_test"] * s["y_std"] + s["y_mean"]
    coords_te = s["coords_test"]
    eligible = [m for m in models
                if target_step in models[m]
                and models[m][target_step].get("y_pred_test") is not None]
    all_res = np.concatenate([
        models[m][target_step]["y_pred_test"] - y_true for m in eligible])
    vmax = float(np.quantile(np.abs(all_res), 0.98))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    n_cols = 3
    n_rows = (len(eligible) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows),
                              constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
    sc = None
    for i, name in enumerate(eligible):
        ax = axes[i]
        res = models[name][target_step]["y_pred_test"] - y_true
        sc = ax.scatter(coords_te[:, 0], coords_te[:, 1], c=res, cmap="RdBu_r",
                          norm=norm, s=4, rasterized=True)
        moran = models[name][target_step]["metrics"]["MoranI"]
        ax.set_title(f"{name}   Moran's I = {moran:.2f}")
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(eligible):]:
        ax.set_visible(False)
    if sc is not None:
        fig.colorbar(sc, ax=axes, label="residual (m³/ha)", shrink=0.6)
    fig.suptitle(f"Test-set residuals in space (n_cells = {target_step!r})", fontsize=11)
    plt.show()
    return fig


def residuals_vs_covariates_figure(models, slice_step_fn, feature_names,
                                   target_step="all", top_k=3):
    """For each eligible model, scatter test residuals vs the top-k |β| covariates."""
    import matplotlib.pyplot as plt
    s = slice_step_fn(target_step)
    y_true = s["y_test"] * s["y_std"] + s["y_mean"]
    X_test = s["X_test"]
    n_expected = 1 + len(feature_names)
    eligible = [m for m in models
                if target_step in models[m]
                and models[m][target_step].get("coefs_samples") is not None
                and models[m][target_step]["coefs_samples"].shape[-1] == n_expected
                and models[m][target_step].get("y_pred_test") is not None]
    if not eligible:
        print("residuals_vs_covariates_figure: no eligible models — run §7 / §8 first")
        return None

    n_rows = len(eligible)
    fig, axes = plt.subplots(n_rows, top_k, figsize=(4 * top_k, 3 * n_rows),
                              constrained_layout=True, sharey=False)
    axes = np.atleast_2d(axes)
    for r, name in enumerate(eligible):
        coefs = models[name][target_step]["coefs_samples"]
        beta_mean = coefs.mean(0)[1:]
        top = np.argsort(-np.abs(beta_mean))[:top_k]
        y_pred = models[name][target_step]["y_pred_test"]
        res = y_pred - y_true
        for c, idx in enumerate(top):
            ax = axes[r, c]
            ax.scatter(X_test[:, idx], res, s=2, alpha=0.2, rasterized=True)
            ax.axhline(0, color="k", lw=0.5, alpha=0.5)
            ax.set_xlabel(f"{feature_names[idx]} (standardised)")
            if c == 0:
                ax.set_ylabel(f"{name}\nresidual (m³/ha)")
            ax.set_title(f"β̂ = {beta_mean[idx]:+.2f}", fontsize=9)
    fig.suptitle(f"Test residuals vs top-{top_k} |β| covariates (n_cells = {target_step!r})",
                  fontsize=11)
    plt.show()
    return fig


def posterior_density_overlay(models, feature_names,
                              methods=("Exact", "SVI", "MCMC"),
                              target_step="all", top_k=4):
    """Marginal posterior of intercept + top-k |β| coefficients across inferences."""
    import matplotlib.pyplot as plt
    def _step(name):
        by_n = models.get(name, {})
        if (target_step in by_n
            and by_n[target_step].get("coefs_samples") is not None
            and by_n[target_step]["coefs_samples"].shape[0] > 1):
            return target_step
        cands = [k for k, v in by_n.items()
                 if v.get("coefs_samples") is not None and v["coefs_samples"].shape[0] > 1]
        return max(cands, key=lambda k: by_n[k]["n_train"]) if cands else None

    present = [(m, _step(m)) for m in methods]
    present = [(m, k) for m, k in present if k is not None]
    if not present:
        print("posterior_density_overlay: need at least one Bayesian model — run §8")
        return None
    ref_name, ref_step = present[0]
    ref_coefs = models[ref_name][ref_step]["coefs_samples"]
    beta_abs = np.abs(ref_coefs.mean(0)[1:])
    top = np.argsort(-beta_abs)[:top_k]
    coef_idx = [0] + [1 + i for i in top]
    coef_lab = ["intercept"] + [feature_names[i] for i in top]

    colors = {"Exact": "C0", "SVI": "C1", "MCMC": "C2"}
    fig, axes = plt.subplots(1, len(coef_idx), figsize=(3 * len(coef_idx), 3),
                              constrained_layout=True)
    for ax, ci, lab in zip(axes, coef_idx, coef_lab):
        for name, step in present:
            coefs = models[name][step]["coefs_samples"]
            ax.hist(coefs[:, ci], bins=40, density=True, alpha=0.45,
                     color=colors.get(name), label=f"{name} (n_cells={step!r})")
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("coef value")
    axes[0].set_ylabel("posterior density")
    axes[-1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Posterior-density overlay across inference methods", fontsize=11)
    plt.show()
    return fig


def posterior_predictive_density_figure(models, slice_step_fn, target_step="all",
                                        n_pp=50, seed=42):
    """Histogram of posterior-predictive y_new vs observed y on the test set.

    Two paths: linear-Gaussian models reconstruct pp from (coefs, sigma);
    non-linear or non-Gaussian models (BNN, cluster variants, LogNormal)
    pass `y_pred_samples_test` directly in original units.
    """
    import matplotlib.pyplot as plt
    s = slice_step_fn(target_step)
    y_true = (s["y_test"] * s["y_std"] + s["y_mean"]).ravel()
    X_test = s["X_test"]
    y_mean_g, y_std_g = s["y_mean"], s["y_std"]

    def _eligible(name):
        if target_step not in models.get(name, {}):
            return False
        p = models[name][target_step]
        c, sgm = p.get("coefs_samples"), p.get("sigma_samples")
        if (c is not None and c.shape[0] > 1
                 and c.shape[-1] == 1 + X_test.shape[1]
                 and sgm is not None):
            return True
        return p.get("y_pred_samples_test") is not None

    eligible = [m for m in models if _eligible(m)]
    if not eligible:
        print("posterior_predictive_density_figure: no eligible models")
        return None
    rng = np.random.default_rng(seed)
    n_cols = 2
    n_rows = (len(eligible) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows),
                              constrained_layout=True, sharex=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, name in zip(axes, eligible):
        p = models[name][target_step]
        coefs = p.get("coefs_samples")
        sigma = p.get("sigma_samples")
        linear_path = (coefs is not None and sigma is not None
                       and coefs.shape[0] > 1
                       and coefs.shape[-1] == 1 + X_test.shape[1])
        if linear_path:
            n_use = min(n_pp, coefs.shape[0])
            draw_idx = rng.choice(coefs.shape[0], size=n_use, replace=False)
            mean_std = coefs[draw_idx, 0:1].T + X_test @ coefs[draw_idx, 1:].T
            sigma_std = sigma[draw_idx][None, :]
            pp_std = mean_std + sigma_std * rng.normal(size=mean_std.shape)
            pp = (pp_std * y_std_g + y_mean_g).ravel()
        else:
            y_samp = p["y_pred_samples_test"]
            n_use = min(n_pp, y_samp.shape[0])
            draw_idx = rng.choice(y_samp.shape[0], size=n_use, replace=False)
            pp = y_samp[draw_idx].ravel()
        ax.hist(y_true, bins=80, density=True, alpha=0.55, label="observed y",            color="C0")
        ax.hist(pp,     bins=80, density=True, alpha=0.55, label="posterior predictive",  color="C3")
        ax.set_title(name); ax.set_xlabel("ΔV (m³/ha)")
        ax.set_xlim(right=500)
        ax.legend(fontsize=8)
    for ax in axes[len(eligible):]:
        ax.set_visible(False)
    fig.suptitle(f"Posterior-predictive vs observed y (n_cells = {target_step!r})", fontsize=11)
    plt.show()
    return fig


# ======================================================================
# Evaluation utilities (used by final_notebook.ipynb §6 onward)
# ======================================================================

from sklearn.neighbors import NearestNeighbors as _NearestNeighbors

# Shared MODELS registry. Every fitted model registers its (scaling-step) result
# here so §9's comparison helpers can read them without bookkeeping in each cell.
# Keyed as MODELS[model_name][n_cells] -> dict with metrics, samples, y_pred, ...
MODELS: dict[str, dict] = {}


def moran_i(values, coords, k: int = 8, n_permutations: int = 999, seed: int = 42):
    """Moran's I on k-NN with row-standardised weights; permutation p-value.

    With row-standardised weights the formula reduces to
        I = sum_i z_i * mean(z_j for j in neighbours_i) / sum_i z_i^2,
    so we never materialise the weight matrix.
    """
    import numpy as _np
    coords = _np.asarray(coords)
    values = _np.asarray(values).ravel()
    nn = _NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    neighbours = idx[:, 1:]  # drop self

    def _i(z):
        num = (z * z[neighbours].mean(axis=1)).sum()
        return float(num / (z @ z))

    z_obs = values - values.mean()
    observed = _i(z_obs)

    rng = _np.random.default_rng(seed)
    perms = _np.fromiter(
        (_i(rng.permutation(z_obs)) for _ in range(n_permutations)),
        dtype=float, count=n_permutations,
    )
    p = (1 + (perms >= observed).sum()) / (1 + n_permutations)
    return observed, float(p)


def evaluate(y_true, y_pred, coords=None) -> dict:
    """Standard regression scorecard + optional Moran's I on residuals."""
    import numpy as _np
    from sklearn.metrics import r2_score as _r2
    y_true = _np.asarray(y_true).ravel()
    y_pred = _np.asarray(y_pred).ravel()
    res = y_pred - y_true
    out = {
        "RMSE": float(_np.sqrt(_np.mean(res ** 2))),
        "MAE":  float(_np.mean(_np.abs(res))),
        "Bias": float(res.mean()),
        "R2":   float(_r2(y_true, y_pred)),
        "Corr": float(_np.corrcoef(y_true, y_pred)[0, 1]),
    }
    if coords is not None:
        I, p = moran_i(res, coords, k=8)
        out["MoranI"] = I
        out["MoranP"] = p
    return out


def register_run(name: str, n_cells, *, metrics: dict, n_train: int, n_test: int,
                 coefs_samples=None, sigma_samples=None, elbo_trace=None,
                 **extras) -> None:
    """Record one (model, n_cells) result for the §9 comparison helpers.

    ``coefs_samples`` is expected to be shape (n_post_samples, n_features + 1),
    with the intercept as column 0. For point models pass shape (1, n_features+1)
    or omit. Extras (e.g. wall_time_s, alpha_v_samples) are stored at the top
    level of the payload.
    """
    MODELS.setdefault(name, {})[n_cells] = {
        "metrics": metrics,
        "n_train": n_train,
        "n_test": n_test,
        "coefs_samples": coefs_samples,
        "sigma_samples": sigma_samples,
        "elbo_trace": elbo_trace,
        **extras,
    }


def all_metrics_df():
    """Flatten MODELS into a long DataFrame for the §9 comparison table."""
    import pandas as _pd
    rows = []
    for model_name, by_n in MODELS.items():
        for n_cells, payload in by_n.items():
            rows.append({"model": model_name, "n_cells": n_cells,
                         "n_train": payload["n_train"], "n_test": payload["n_test"],
                         **payload["metrics"]})
    return _pd.DataFrame(rows)

