"""Helpers backing ``final_notebook.ipynb``.

Contents:
  §3  BK-cell splits       — build_splits / get_train_bk / load_or_build_splits / make_slicer
  §4  factor model         — 2-factor Bayesian FA on six lidar views (fit + scoring)
  §6  evaluation registry  — moran_i / evaluate / register_run / MODELS
  §8  Bayesian regression  — Pyro models + exact / SVI / NUTS fits
  §8.x per-model fits      — fit_lognormal_blr / fit_cluster_intercept / ... / run_across_scaling
  §9  comparison figures   — scaling, forest, calibration, spatial maps, etc.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS, Predictive
from pyro.infer.autoguide import AutoNormal, AutoDiagonalNormal
from pyro.optim import Adam


# ======================================================================
# §3 — BK-cell-disjoint train/test splits
# ======================================================================

def build_splits(df: pd.DataFrame, test_frac: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    all_bk = sorted(df["BK"].unique().tolist())
    rng.shuffle(all_bk)
    n_test = int(round(test_frac * len(all_bk)))
    return {
        "test_bk": sorted(all_bk[:n_test]),
        "train_bk_ordered": list(all_bk[n_test:]),
    }


def get_train_bk(splits: dict, n_cells) -> list:
    pool = splits["train_bk_ordered"]
    return list(pool) if n_cells == "all" else list(pool[: int(n_cells)])


def load_or_build_splits(df: pd.DataFrame, splits_json: Path,
                         test_frac: float, seed: int) -> dict:
    splits_json = Path(splits_json)
    if splits_json.exists():
        splits = json.loads(splits_json.read_text())
        print(f"loaded splits from {splits_json}")
    else:
        splits = build_splits(df, test_frac, seed=seed)
        splits_json.write_text(json.dumps(splits))
        print(f"built and saved splits → {splits_json}")
    return splits


def make_slicer(X_all, y_all, coords_all, bk_all, splits):
    """Return a slice_step(n_cells, X=None) closure that yields the
    standardised train/test slice at one scaling step."""
    def slice_step(n_cells, X=None):
        X_ = X_all if X is None else X
        train_bk = set(get_train_bk(splits, n_cells))
        test_bk = set(splits["test_bk"])
        tr = np.isin(bk_all, list(train_bk))
        te = np.isin(bk_all, list(test_bk))

        Xtr, ytr = X_[tr], y_all[tr]
        Xte, yte = X_[te], y_all[te]

        x_mean = Xtr.mean(axis=0)
        x_std = Xtr.std(axis=0) + 1e-8
        y_mean, y_std = float(ytr.mean()), float(ytr.std() + 1e-8)

        return {
            "X_train": (Xtr - x_mean) / x_std,
            "y_train": (ytr - y_mean) / y_std,
            "X_test":  (Xte - x_mean) / x_std,
            "y_test":  (yte - y_mean) / y_std,
            "coords_test": coords_all[te],
            "y_mean": y_mean, "y_std": y_std,
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
        }
    return slice_step


# ======================================================================
# §4 — Factor analysis on six collinear lidar views
# ======================================================================

# Anchor order matters for identification:
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
# §6 — Evaluation utilities + MODELS registry
# ======================================================================

from sklearn.neighbors import NearestNeighbors as _NearestNeighbors

# Every fitted model registers its (scaling-step) result here so §9's comparison
# helpers can read them without bookkeeping in each cell.
# Keyed as MODELS[model_name][n_cells] -> dict with metrics, samples, y_pred, ...
MODELS: dict[str, dict] = {}


def moran_i(values, coords, k: int = 8, n_permutations: int = 999, seed: int = 42):
    """Moran's I on k-NN with row-standardised weights; permutation p-value.

    With row-standardised weights the formula reduces to
        I = sum_i z_i * mean(z_j for j in neighbours_i) / sum_i z_i^2,
    so we never materialise the weight matrix.
    """
    coords = np.asarray(coords)
    values = np.asarray(values).ravel()
    nn = _NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    neighbours = idx[:, 1:]  # drop self

    def _i(z):
        num = (z * z[neighbours].mean(axis=1)).sum()
        return float(num / (z @ z))

    z_obs = values - values.mean()
    observed = _i(z_obs)

    rng = np.random.default_rng(seed)
    perms = np.fromiter(
        (_i(rng.permutation(z_obs)) for _ in range(n_permutations)),
        dtype=float, count=n_permutations,
    )
    p = (1 + (perms >= observed).sum()) / (1 + n_permutations)
    return observed, float(p)


def evaluate(y_true, y_pred, coords=None) -> dict:
    """Standard regression scorecard + optional Moran's I on residuals."""
    from sklearn.metrics import r2_score as _r2
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    res = y_pred - y_true
    out = {
        "RMSE": float(np.sqrt(np.mean(res ** 2))),
        "MAE":  float(np.mean(np.abs(res))),
        "Bias": float(res.mean()),
        "R2":   float(_r2(y_true, y_pred)),
        "Corr": float(np.corrcoef(y_true, y_pred)[0, 1]),
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

    Identity-link Gaussian BLR contract (most models):
      ``coefs_samples`` has shape (n_post_samples, 1 + n_features), with the
      intercept at column 0; ``sigma_samples`` has shape (n_post_samples,).
      §9 helpers reconstruct PPC as alpha + X beta + sigma * N(0,1) on
      standardised y and then un-standardise.

    Non-identity-link or non-linear models (LogNormal, BNN, SVI-clip0, cluster
    variants, anything with a custom likelihood or fabricated PPC):
      Register ``y_pred_samples_test`` as a (n_post_samples, n_test) array of
      posterior-predictive draws *on the original y scale (m³/ha)*. The §9
      density figure and calibration plot prefer ``y_pred_samples_test`` over
      the linear reconstruction whenever present.

    For point models (OLS / HGBT) pass coefs_samples shape (1, 1+n_features)
    or omit; they are skipped by the Bayesian-only diagnostics.

    ``sigma_samples`` units are NOT cross-model comparable: it is the SD on
    standardised y for the Gaussian-BLR family and the cluster variants, but
    the SD on log-y for LogNormal. Do not plot it across models directly.

    Extras (wall_time_s, alpha_v_samples, beta_v_samples, y_pred_samples_test,
    ...) are stored at the top level of the payload.
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
    rows = []
    for model_name, by_n in MODELS.items():
        for n_cells, payload in by_n.items():
            rows.append({"model": model_name, "n_cells": n_cells,
                         "n_train": payload["n_train"], "n_test": payload["n_test"],
                         **payload["metrics"]})
    return pd.DataFrame(rows)


def save_model_runs(path, names):
    """Pickle the MODELS payloads for `names` to `path` — caches expensive fits
    (e.g. NUTS) so downstream work can reload them instead of refitting."""
    path = Path(path)
    subset = {name: MODELS[name] for name in names if name in MODELS}
    path.write_bytes(pickle.dumps(subset))
    return path


def load_model_runs(path):
    """Load pickled model payloads from `path` and merge them into MODELS.
    Returns the list of model names loaded."""
    subset = pickle.loads(Path(path).read_bytes())
    MODELS.update(subset)
    return list(subset)


# ======================================================================
# §8 — Bayesian regression (Pyro models + exact / SVI / NUTS fits)
# ======================================================================

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
# §8.x — Spatial-lag utilities and per-model SVI fits
# ======================================================================

def compute_knn_lag(coords, features, k):
    """Mean of each pixel's k nearest spatial neighbours (self excluded)."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    return features[idx[:, 1:]].mean(axis=1)


def compute_WX(coords_ref, X_ref, coords_query, k, exclude_self=False):
    """Mean standardised feature of k nearest neighbours from coords_ref."""
    from sklearn.neighbors import NearestNeighbors
    k_eff = min(k + int(exclude_self), len(X_ref))
    nn = NearestNeighbors(n_neighbors=k_eff).fit(coords_ref)
    _, idx = nn.kneighbors(coords_query)
    if exclude_self:
        idx = idx[:, 1:]
    return X_ref[idx].mean(axis=1)


def fit_lognormal_blr(s: dict, *, log_eps: float = 1.0,
                      n_steps: int = 2000, n_posterior: int = 1000,
                      lr: float = 0.01, seed: int = 42) -> dict:
    """Fit the §6.1 Gaussian BLR in standardised log-space and back-transform.

    Returns a dict with metrics, fit summaries, and a ``register_kwargs`` dict
    ready to splat into :func:`register_run`.
    """
    y_tr_orig = s["y_train"] * s["y_std"] + s["y_mean"]
    y_te_orig = s["y_test"]  * s["y_std"] + s["y_mean"]

    log_y_tr   = np.log(np.maximum(y_tr_orig, log_eps))
    log_y_mean = float(log_y_tr.mean())
    log_y_sd   = float(log_y_tr.std() + 1e-8)
    log_y_tr_s = (log_y_tr - log_y_mean) / log_y_sd

    X_tr_t = torch.tensor(s["X_train"], dtype=torch.float32)
    y_tr_t = torch.tensor(log_y_tr_s,   dtype=torch.float32)

    pyro.clear_param_store()
    pyro.set_rng_seed(seed); torch.manual_seed(seed)
    guide = AutoDiagonalNormal(blr_pyro_model)
    svi   = SVI(blr_pyro_model, guide, Adam({"lr": lr}), loss=Trace_ELBO())

    elbo = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        elbo.append(svi.step(X_tr_t, y_tr_t))
    wall = time.perf_counter() - t0

    predictive = Predictive(blr_pyro_model, guide=guide,
                            num_samples=n_posterior,
                            return_sites=("alpha", "beta", "sigma"))
    post = {k: v.detach().numpy() for k, v in predictive(X_tr_t, y_tr_t).items()}

    alpha_post   = post["alpha"].reshape(-1)
    beta_post    = post["beta"].reshape(len(alpha_post), -1)
    sigma_post_s = post["sigma"].reshape(-1)
    sigma_post   = sigma_post_s * log_y_sd

    log_mean_s = alpha_post[:, None] + beta_post @ s["X_test"].T
    log_mean   = log_mean_s * log_y_sd + log_y_mean

    rng_pp  = np.random.default_rng(seed)
    eps_log = rng_pp.normal(size=log_mean.shape) * sigma_post[:, None]
    y_pred_samples_test = np.exp(log_mean + eps_log)
    # E[y | log_mean, sigma] = exp(log_mean + sigma^2/2); average over (α,β,σ).
    y_hat = np.exp(log_mean + 0.5 * sigma_post[:, None] ** 2).mean(axis=0)

    m = evaluate(y_te_orig, y_hat, coords=s["coords_test"])
    return {
        "metrics": m,
        "log_y_mean": log_y_mean,
        "log_y_std": log_y_sd,
        "sigma_log_mean": float(sigma_post.mean()),
        "final_elbo": float(elbo[-1]),
        "register_kwargs": dict(
            metrics=m, n_train=s["n_train"], n_test=s["n_test"],
            coefs_samples=None, sigma_samples=sigma_post,
            elbo_trace=elbo,
            y_pred_test=y_hat, y_pred_samples_test=y_pred_samples_test,
            wall_time_s=wall,
        ),
    }


def fit_cluster_intercept(s: dict, K: int, model_fn, *,
                          n_steps: int = 2000, n_posterior: int = 1000,
                          lr: float = 0.01, seed: int = 42) -> dict:
    """Fit a cluster-intercept Pyro model at given K on slice s.

    k-means on standardised training features assigns cluster ids; test pixels
    go to their nearest centroid in feature space.
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=K, random_state=seed, n_init=5)
    km.fit(s["X_train"])
    ids_tr = km.predict(s["X_train"])
    ids_te = km.predict(s["X_test"])

    X_tr_t   = torch.tensor(s["X_train"], dtype=torch.float32)
    y_tr_t   = torch.tensor(s["y_train"], dtype=torch.float32)
    ids_tr_t = torch.tensor(ids_tr, dtype=torch.long)

    pyro.clear_param_store()
    pyro.set_rng_seed(seed); torch.manual_seed(seed)
    guide = AutoDiagonalNormal(model_fn)
    svi   = SVI(model_fn, guide, Adam({"lr": lr}), loss=Trace_ELBO())

    elbo = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        elbo.append(svi.step(X_tr_t, ids_tr_t, K, y_tr_t))
    wall = time.perf_counter() - t0

    predictive = Predictive(model_fn, guide=guide, num_samples=n_posterior,
                            return_sites=("alpha", "beta", "sigma_a", "sigma"))
    post = {k: v.detach().numpy() for k, v in predictive(
        X_tr_t, ids_tr_t, K, y_tr_t).items()}

    alpha_post  = post["alpha"].reshape(-1, K)
    beta_post   = post["beta"].reshape(-1, X_tr_t.shape[1])
    sigma_post  = post["sigma"].reshape(-1)
    sigma_a_pst = post["sigma_a"].reshape(-1)

    rng_pp   = np.random.default_rng(seed)
    mean_std = alpha_post[:, ids_te] + beta_post @ s["X_test"].T
    eps      = rng_pp.normal(size=mean_std.shape) * sigma_post[:, None]
    y_pred_samples_test = (mean_std + eps) * s["y_std"] + s["y_mean"]

    y_hat_std = mean_std.mean(axis=0)
    y_hat     = y_hat_std * s["y_std"] + s["y_mean"]
    y_true    = s["y_test"] * s["y_std"] + s["y_mean"]
    m         = evaluate(y_true, y_hat, coords=s["coords_test"])

    return {
        "metrics": m,
        "sigma_a_mean": float(sigma_a_pst.mean()),
        "final_elbo": float(elbo[-1]),
        "register_kwargs": dict(
            metrics=m, n_train=s["n_train"], n_test=s["n_test"],
            coefs_samples=None, sigma_samples=sigma_post,
            elbo_trace=elbo, sigma_a_samples=sigma_a_pst,
            y_pred_test=y_hat, y_pred_samples_test=y_pred_samples_test,
            wall_time_s=wall,
        ),
    }


def fit_cluster_featurelag(s: dict, WX_train, WX_test, ids_tr, ids_te,
                           K: int, model_fn, *,
                           n_steps: int = 2000, n_posterior: int = 1000,
                           lr: float = 0.01, seed: int = 42) -> dict:
    """Fit the cluster-intercept + spatial-feature-lag model on slice s.

    ``WX_*`` (spatial feature lags) and ``ids_*`` (cluster assignments) are
    pre-computed by the caller, since that setup is model-specific.
    """
    X_tr_t   = torch.tensor(s["X_train"], dtype=torch.float32)
    WX_tr_t  = torch.tensor(WX_train,     dtype=torch.float32)
    y_tr_t   = torch.tensor(s["y_train"], dtype=torch.float32)
    ids_tr_t = torch.tensor(ids_tr,       dtype=torch.long)

    pyro.clear_param_store()
    pyro.set_rng_seed(seed); torch.manual_seed(seed)
    guide = AutoDiagonalNormal(model_fn)
    svi   = SVI(model_fn, guide, Adam({"lr": lr}), loss=Trace_ELBO())

    elbo = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        elbo.append(svi.step(X_tr_t, WX_tr_t, ids_tr_t, K, y_tr_t))
    wall = time.perf_counter() - t0

    predictive = Predictive(
        model_fn, guide=guide, num_samples=n_posterior,
        return_sites=("alpha", "beta", "gamma", "sigma_a", "sigma"))
    post = {k: v.detach().numpy() for k, v in predictive(
        X_tr_t, WX_tr_t, ids_tr_t, K, y_tr_t).items()}

    alpha_post  = post["alpha"].reshape(-1, K)
    beta_post   = post["beta"].reshape(-1, X_tr_t.shape[1])
    gamma_post  = post["gamma"].reshape(-1, X_tr_t.shape[1])
    sigma_post  = post["sigma"].reshape(-1)
    sigma_a_pst = post["sigma_a"].reshape(-1)
    gamma_norm  = float(np.linalg.norm(gamma_post, axis=1).mean())

    mean_std = (alpha_post[:, ids_te] + beta_post @ s["X_test"].T
                + gamma_post @ WX_test.T)
    rng_pp   = np.random.default_rng(seed)
    eps      = rng_pp.normal(size=mean_std.shape) * sigma_post[:, None]
    y_pred_samples_test = (mean_std + eps) * s["y_std"] + s["y_mean"]

    y_hat_std = mean_std.mean(axis=0)
    y_hat     = y_hat_std * s["y_std"] + s["y_mean"]
    y_true    = s["y_test"] * s["y_std"] + s["y_mean"]
    m         = evaluate(y_true, y_hat, coords=s["coords_test"])

    return {
        "metrics": m,
        "sigma_a_mean": float(sigma_a_pst.mean()),
        "gamma_norm": gamma_norm,
        "final_elbo": float(elbo[-1]),
        "register_kwargs": dict(
            metrics=m, n_train=s["n_train"], n_test=s["n_test"],
            coefs_samples=None, sigma_samples=sigma_post,
            elbo_trace=elbo, sigma_a_samples=sigma_a_pst,
            gamma_norm=gamma_norm,
            y_pred_test=y_hat, y_pred_samples_test=y_pred_samples_test,
            wall_time_s=wall,
        ),
    }


def fit_bnn(s: dict, model_cls, *, hidden_dim: int = 16,
            n_steps: int = 2000, n_posterior: int = 300,
            lr: float = 0.005, seed: int = 42) -> dict:
    """Fit a Bayesian neural network (a PyroModule class) on slice s."""
    X_tr_t = torch.tensor(s["X_train"], dtype=torch.float32)
    y_tr_t = torch.tensor(s["y_train"], dtype=torch.float32)
    X_te_t = torch.tensor(s["X_test"],  dtype=torch.float32)

    pyro.clear_param_store()
    pyro.set_rng_seed(seed); torch.manual_seed(seed)
    model = model_cls(input_dim=X_tr_t.shape[1], hidden_dim=hidden_dim)
    guide = AutoDiagonalNormal(model)
    svi   = SVI(model, guide, Adam({"lr": lr}), loss=Trace_ELBO())

    elbo = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        elbo.append(svi.step(X_tr_t, y_tr_t))
    wall = time.perf_counter() - t0

    predictive = Predictive(model, guide=guide, num_samples=n_posterior,
                            return_sites=("mean", "y", "sigma"))
    with torch.no_grad():
        pred = predictive(X_te_t)

    y_hat_std       = pred["mean"].mean(dim=0).reshape(-1).numpy()
    y_pred_samp_std = pred["y"].numpy()
    sigma_post_std  = pred["sigma"].reshape(-1).numpy()

    y_hat               = y_hat_std       * s["y_std"] + s["y_mean"]
    y_pred_samples_test = y_pred_samp_std * s["y_std"] + s["y_mean"]
    y_true              = s["y_test"] * s["y_std"] + s["y_mean"]
    m = evaluate(y_true, y_hat, coords=s["coords_test"])

    return {
        "metrics": m,
        "sigma_mean_std": float(sigma_post_std.mean()),
        "final_elbo": float(elbo[-1]),
        "register_kwargs": dict(
            metrics=m, n_train=s["n_train"], n_test=s["n_test"],
            coefs_samples=None, sigma_samples=sigma_post_std,
            elbo_trace=elbo,
            y_pred_test=y_hat, y_pred_samples_test=y_pred_samples_test,
            wall_time_s=wall,
        ),
    }


def run_across_scaling(name, scaling_grid, slice_fn, fit_fn, *,
                       extra_cols=None):
    """Drive a model across the scaling grid: slice → fit → register_run.

    ``fit_fn(n_cells, s)`` must return a dict with ``register_kwargs`` and
    ``metrics``; any other top-level scalar entries become summary-row columns.
    ``extra_cols`` is merged into every row (e.g. fixed hyper-parameters).
    Returns the per-step summary DataFrame.
    """
    rows = []
    for n_cells in scaling_grid:
        s = slice_fn(n_cells)
        r = fit_fn(n_cells, s)
        register_run(name, n_cells, **r["register_kwargs"])
        extra = {k: v for k, v in r.items()
                 if k not in ("register_kwargs", "metrics")}
        rows.append({"n_cells": n_cells, "n_train": s["n_train"],
                     "n_test": s["n_test"], **(extra_cols or {}),
                     **extra, **r["metrics"]})
    return pd.DataFrame(rows)


# ======================================================================
# §9 — Comparison figures
# ======================================================================

def posterior_predictive_coverage(models, slice_step_fn, name, n_cells,
                                  alpha_levels, seed=42):
    """Empirical coverage at every nominal credible level for one (model, step).

    Resolution order on each payload:
      1. If ``y_pred_samples_test`` is registered (S × n_test, original-y units), use it.
      2. Else if the payload is identity-link Gaussian BLR (``coefs_samples`` shape
         (S, 1+d) with S > 1 and ``sigma_samples`` set), reconstruct
         pp_std = α + Xβ + σ·N(0,1) and un-standardise.
      3. Else return None — the calibration plot drops this model.
    """
    s = slice_step_fn(n_cells)
    payload = models[name][n_cells]
    y_true = s["y_test"] * s["y_std"] + s["y_mean"]

    if payload.get("y_pred_samples_test") is not None:
        pp = np.asarray(payload["y_pred_samples_test"])
    else:
        coefs = payload.get("coefs_samples")
        if coefs is None or coefs.shape[0] < 2:
            return None
        n_expected = 1 + s["X_test"].shape[1]
        if coefs.shape[-1] != n_expected:
            return None
        sigma = payload.get("sigma_samples")
        if sigma is None:
            return None
        alpha_samples = coefs[:, 0]
        beta_samples  = coefs[:, 1:]
        mean_std = alpha_samples[:, None] + beta_samples @ s["X_test"].T
        sigma_std = sigma[:, None]
        rng = np.random.default_rng(seed)
        pp_std = mean_std + sigma_std * rng.normal(size=mean_std.shape)
        pp = pp_std * s["y_std"] + s["y_mean"]

    coverage_curve = []
    for a in alpha_levels:
        lo = np.quantile(pp, (1 - a) / 2, axis=0)
        hi = np.quantile(pp, (1 + a) / 2, axis=0)
        coverage_curve.append(float(((y_true >= lo) & (y_true <= hi)).mean()))
    return np.array(coverage_curve)


def calibration_figure(models, slice_step_fn,
                       candidate_models=None, alpha_levels=None, seed=42):
    """Empirical vs nominal posterior-predictive coverage, one line per Bayesian model.

    By default every registered model is tried; models with no predictive
    distribution (OLS / HGBT) yield ``None`` from posterior_predictive_coverage
    and are skipped. Pass ``candidate_models`` to restrict the set explicitly.
    """
    import matplotlib.pyplot as plt
    if alpha_levels is None:
        alpha_levels = np.arange(0.1, 1.0, 0.1)
    if candidate_models is None:
        candidate_models = list(models)
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
                  "(point models OLS / HGBT cannot appear — no predictive distribution)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.show()


# ======================================================================
# §9.x — Compact report figures (models as series on a shared axis)
# ======================================================================

def metric_dotplot(models, metric="RMSE", target_step="all",
                   ceiling_models=("OLS", "HGBT")):
    """Horizontal dot plot ranking every model on one metric at `target_step`.

    Covers both the §7.1 point-accuracy ranking (metric="RMSE") and the §7.4
    spatial-leakage ranking (metric="MoranI"). ``ceiling_models`` are drawn as
    vertical reference lines, not dots — the baselines everything else is
    judged against.
    """
    import matplotlib.pyplot as plt
    rows = []
    for name, by_n in models.items():
        if name in ceiling_models or target_step not in by_n:
            continue
        val = by_n[target_step]["metrics"].get(metric)
        if val is not None:
            rows.append((name, val))
    if not rows:
        print(f"metric_dotplot: no models with {metric!r} at {target_step!r}")
        return None
    rows.sort(key=lambda kv: kv[1])
    names = [r[0] for r in rows]
    vals  = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 0.42 * len(rows) + 1.2),
                           constrained_layout=True)
    y = np.arange(len(rows))
    ax.scatter(vals, y, s=70, color="C0", zorder=3)
    for offset, ceil in enumerate(ceiling_models):
        if ceil in models and target_step in models[ceil]:
            cv = models[ceil][target_step]["metrics"].get(metric)
            if cv is not None:
                ax.axvline(cv, ls="--", color=f"C{offset + 1}", alpha=0.8,
                           label=f"{ceil} = {cv:.2f}")
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlabel(metric)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title(f"{metric} by model  (n_cells = {target_step!r})")
    if any(c in models for c in ceiling_models):
        ax.legend(fontsize=8, loc="best")
    plt.show()


def inference_forest_plot(models, feature_names,
                          methods=("Exact", "SVI", "MCMC"),
                          point_models=("OLS",), target_step="all", top_k=6):
    """Forest plot restricted to the inference-method comparison.

    Posterior 90% CIs for `methods`, point estimates (diamonds, no bars) for
    `point_models`, on the intercept + top-k |β| features ranked by the first
    available method.
    """
    import matplotlib.pyplot as plt
    feat_full = ["intercept"] + list(feature_names)
    n_expected = len(feat_full)

    def _coefs(name):
        by_n = models.get(name, {})
        if target_step in by_n:
            c = by_n[target_step].get("coefs_samples")
            if c is not None and c.shape[-1] == n_expected:
                return c
        return None

    method_coefs = [(m, _coefs(m)) for m in methods]
    method_coefs = [(m, c) for m, c in method_coefs if c is not None]
    if not method_coefs:
        print(f"inference_forest_plot: no methods have coefs at {target_step!r}")
        return None

    ref = method_coefs[0][1]
    top = np.argsort(-np.abs(ref.mean(axis=0)[1:]))[:top_k]
    sel = [0] + [1 + i for i in top]            # intercept first
    sel_labels = [feat_full[i] for i in sel]

    point_coefs = [(n, _coefs(n)) for n in point_models]
    point_coefs = [(n, c) for n, c in point_coefs if c is not None]

    n_series = len(method_coefs) + len(point_coefs)
    offsets = np.linspace(-0.3, 0.3, n_series) if n_series > 1 else [0.0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_series, 2)))

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(sel) + 1.5),
                           constrained_layout=True)
    yb = np.arange(len(sel))
    for i in range(0, len(sel), 2):
        ax.axhspan(i - 0.5, i + 0.5, color="gray", alpha=0.07, zorder=0)

    idx = 0
    for name, c in method_coefs:
        means = c.mean(axis=0)[sel]
        lo = np.quantile(c, 0.05, axis=0)[sel]
        hi = np.quantile(c, 0.95, axis=0)[sel]
        ax.errorbar(means, yb + offsets[idx], xerr=[means - lo, hi - means],
                    fmt="o", capsize=3, color=colors[idx], label=name,
                    alpha=0.9, zorder=2)
        idx += 1
    for name, c in point_coefs:
        ax.scatter(c.mean(axis=0)[sel], yb + offsets[idx], marker="D",
                   color=colors[idx], label=f"{name} (point)", zorder=2)
        idx += 1

    ax.axvline(0, color="k", ls=":", alpha=0.5)
    ax.set_yticks(yb); ax.set_yticklabels(sel_labels)
    ax.set_ylim(len(sel) - 0.5, -0.5)
    ax.set_xlabel("standardised coefficient (intercept + β)")
    ax.set_title(f"Inference-method agreement (n_cells = {target_step!r})")
    ax.legend(fontsize=8, loc="best")
    plt.show()


def ppc_density_overlay(models, slice_step_fn, names, target_step="all",
                        n_pp=80, seed=42):
    """Overlay posterior-predictive densities for `names` against observed y on
    a single panel — compact replacement for the per-model PPC histogram grid.

    Each model's predictive is resolved as in posterior_predictive_coverage:
    registered ``y_pred_samples_test`` if present, else identity-link
    Gaussian-BLR reconstruction from ``coefs_samples`` + ``sigma_samples``.
    """
    import matplotlib.pyplot as plt
    s = slice_step_fn(target_step)
    y_true = s["y_test"] * s["y_std"] + s["y_mean"]
    rng = np.random.default_rng(seed)

    def _pp(name):
        by_n = models.get(name, {})
        if target_step not in by_n:
            return None
        p = by_n[target_step]
        if p.get("y_pred_samples_test") is not None:
            arr = np.asarray(p["y_pred_samples_test"])
        else:
            c, sgm = p.get("coefs_samples"), p.get("sigma_samples")
            if c is None or c.shape[0] < 2 or sgm is None:
                return None
            if c.shape[-1] != 1 + s["X_test"].shape[1]:
                return None
            mean_std = c[:, 0:1] + c[:, 1:] @ s["X_test"].T
            pp_std = mean_std + sgm[:, None] * rng.normal(size=mean_std.shape)
            arr = pp_std * s["y_std"] + s["y_mean"]
        k = min(n_pp, arr.shape[0])
        return arr[rng.choice(arr.shape[0], size=k, replace=False)].ravel()

    from scipy.stats import gaussian_kde

    series = []
    for name in names:
        pp = _pp(name)
        if pp is None:
            print(f"ppc_density_overlay: skipping {name!r} (no predictive)")
            continue
        series.append((name, pp))
    if not series:
        print("ppc_density_overlay: no eligible models")
        return None

    # Window spans the central 99% of observed y AND of every model's
    # predictive, so the left tails of the (near-)Gaussian predictives are
    # captured; a heavy right tail (LogNormal) is bounded by its own 99.5th
    # percentile rather than its extreme max.
    qs = [np.quantile(y_true, [0.005, 0.995])]
    qs += [np.quantile(pp, [0.005, 0.995]) for _, pp in series]
    lo = min(q[0] for q in qs)
    hi = max(q[1] for q in qs)
    pad = 0.05 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 400)

    def _kde(x):
        # subsample for speed — KDE shape is stable well below the full sample
        if x.size > 40000:
            x = rng.choice(x, size=40000, replace=False)
        return gaussian_kde(x)(grid)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.fill_between(grid, _kde(y_true), color="0.6", alpha=0.4,
                    label="observed y")
    for i, (name, pp) in enumerate(series):
        ax.plot(grid, _kde(pp), color=f"C{i}", linewidth=1.8, label=name)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("ΔV (m³/ha)"); ax.set_ylabel("density")
    ax.set_title(f"Posterior-predictive vs observed (n_cells = {target_step!r})")
    ax.legend(fontsize=8)
    plt.show()


def residual_comparison_maps(models, slice_step_fn, left, right,
                             target_step="all"):
    """Two residual maps side by side — e.g. a plain linear model (`left`) vs
    the cluster-intercept model (`right`) — sharing one diverging colour scale.
    The before/after of spatial residual structure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    s = slice_step_fn(target_step)
    y_true = s["y_test"] * s["y_std"] + s["y_mean"]
    coords = s["coords_test"]

    panels = []
    for name in (left, right):
        by_n = models.get(name, {})
        if target_step in by_n and by_n[target_step].get("y_pred_test") is not None:
            panels.append((name, by_n[target_step]))
        else:
            print(f"residual_comparison_maps: {name!r} has no y_pred_test at "
                  f"{target_step!r}")
    if len(panels) != 2:
        return None

    res = [pl["y_pred_test"] - y_true for _, pl in panels]
    vmax = float(np.quantile(np.abs(np.concatenate(res)), 0.98))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    sc = None
    for ax, (name, pl), r in zip(axes, panels, res):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=r, cmap="RdBu_r",
                        norm=norm, s=5, rasterized=True)
        moran = pl["metrics"].get("MoranI")
        ax.set_title(name + (f"   Moran's I = {moran:.3f}"
                             if moran is not None else ""))
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=axes, label="residual (m³/ha)", shrink=0.7)
    fig.suptitle(f"Residual structure: {left} vs {right} "
                 f"(n_cells = {target_step!r})", fontsize=11)
    plt.show()


def walltime_figure(models):
    """Wall-clock time per fit vs training-set size, log-y, one line per model."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for name, by_n in models.items():
        items = sorted(by_n.items(), key=lambda kv: kv[1]["n_train"])
        xs, ys = [], []
        for _, pl in items:
            w = pl.get("wall_time_s")
            if w is not None:
                xs.append(pl["n_train"]); ys.append(w)
        if xs:
            ax.plot(xs, ys, marker="o", label=name, alpha=0.85)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training pixels"); ax.set_ylabel("wall time (s)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="best")
    ax.set_title("Wall-clock compute vs training-set size")
    plt.show()


def tradeoff_scatter(models, x="RMSE", y="MoranI", target_step="all"):
    """2-D trade-off scatter: one labelled point per model. Shows that point
    accuracy (`x`) barely separates the models while another axis (`y`) does —
    the synthesis figure for the whole comparison.
    """
    import matplotlib.pyplot as plt
    pts = []
    for name, by_n in models.items():
        if target_step not in by_n:
            continue
        m = by_n[target_step]["metrics"]
        xv, yv = m.get(x), m.get(y)
        if xv is not None and yv is not None:
            pts.append((name, xv, yv))
    if not pts:
        print(f"tradeoff_scatter: no models with both {x!r} and {y!r}")
        return None
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for name, xv, yv in pts:
        ax.scatter(xv, yv, s=70, zorder=3)
        ax.annotate(name, (xv, yv), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Model trade-off: {x} vs {y}  (n_cells = {target_step!r})")
    plt.show()
