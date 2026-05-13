"""Helpers backing ``final_notebook.ipynb``.

Contents:
  §4  factor model         — 2-factor Bayesian FA on six lidar views (fit + scoring)
  §6  evaluation registry  — moran_i / evaluate / register_run / MODELS
  §8  Bayesian regression  — Pyro models + exact / SVI / NUTS fits
  §9  comparison figures   — scaling, forest, calibration, spatial maps, etc.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS, Predictive
from pyro.infer.autoguide import AutoNormal, AutoDiagonalNormal
from pyro.optim import Adam


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

    Non-identity-link or non-linear models (LogNormal, BNN, SAR, SVI-clip0,
    anything with a custom likelihood):
      Register ``pp_samples`` as a (n_post_samples, n_test) array of
      posterior-predictive draws *on the original y scale (m³/ha)*. The §9
      density figure and calibration plot prefer pp_samples over the linear
      reconstruction whenever present.

    For point models (OLS / GBM) pass coefs_samples shape (1, 1+n_features)
    or omit; they are skipped by the Bayesian-only diagnostics.

    Extras (wall_time_s, alpha_v_samples, beta_v_samples, pp_samples, ...)
    are stored at the top level of the payload.
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
# §9 — Comparison figures
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


def posterior_predictive_coverage(models, slice_step_fn, name, n_cells,
                                  alpha_levels, seed=42):
    """Empirical coverage at every nominal credible level for one (model, step).

    Resolution order on each payload:
      1. If ``pp_samples`` is registered (S × n_test, original-y units), use it.
      2. Else if the payload is identity-link Gaussian BLR (``coefs_samples`` shape
         (S, 1+d) with S > 1 and ``sigma_samples`` set), reconstruct
         pp_std = α + Xβ + σ·N(0,1) and un-standardise.
      3. Else return None — the calibration plot drops this model.

    Heteroscedastic BLR is special-cased: its σ varies with X via
    (alpha_v_samples, beta_v_samples) instead of a single sigma_samples vector.
    """
    s = slice_step_fn(n_cells)
    payload = models[name][n_cells]
    y_true = s["y_test"] * s["y_std"] + s["y_mean"]

    if payload.get("pp_samples") is not None:
        pp = np.asarray(payload["pp_samples"])
    else:
        coefs = payload.get("coefs_samples")
        if coefs is None or coefs.shape[0] < 2:
            return None
        n_expected = 1 + s["X_test"].shape[1]
        if coefs.shape[-1] != n_expected:
            return None
        sigma = payload.get("sigma_samples")
        alpha_samples = coefs[:, 0]
        beta_samples  = coefs[:, 1:]
        mean_std = alpha_samples[:, None] + beta_samples @ s["X_test"].T
        if name == "Heteroscedastic":
            alpha_v = payload.get("alpha_v_samples")
            beta_v  = payload.get("beta_v_samples")
            if alpha_v is None or beta_v is None:
                return None
            log_var  = alpha_v[:, None] + beta_v @ s["X_test"].T
            sigma_std = np.exp(0.5 * log_var)
        elif sigma is not None:
            sigma_std = sigma[:, None]
        else:
            return None
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
    # Zoom window: top-left 60% x 60% of the AOI (anchored at x_min, y_max).
    x_min, x_max = coords_te[:, 0].min(), coords_te[:, 0].max()
    y_min, y_max = coords_te[:, 1].min(), coords_te[:, 1].max()
    dx, dy = 0.6 * (x_max - x_min), 0.6 * (y_max - y_min)
    xlim = (x_min, x_min + dx)
    ylim = (y_max - dy, y_max)

    sc = None
    for i, name in enumerate(eligible):
        ax = axes[i]
        res = models[name][target_step]["y_pred_test"] - y_true
        sc = ax.scatter(coords_te[:, 0], coords_te[:, 1], c=res, cmap="RdBu_r",
                          norm=norm, s=4, rasterized=True)
        moran = models[name][target_step]["metrics"]["MoranI"]
        ax.set_title(f"{name}   Moran's I = {moran:.2f}")
        ax.set_aspect("equal")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(eligible):]:
        ax.set_visible(False)
    if sc is not None:
        fig.colorbar(sc, ax=axes, label="residual (m³/ha)", shrink=0.6)
    fig.suptitle(f"Test-set residuals in space (n_cells = {target_step!r})", fontsize=11)
    plt.show()


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


def posterior_predictive_density_figure(models, slice_step_fn, target_step="all",
                                        n_pp=50, seed=42):
    """Histogram of posterior-predictive y_new vs observed y on the test set.

    Resolution order per model (same as posterior_predictive_coverage):
      1. ``pp_samples`` array on the original-y scale, if registered — used directly.
      2. Identity-link Gaussian BLR reconstruction from
         ``coefs_samples`` (S, 1+d) and ``sigma_samples`` (S,) — for models whose
         likelihood is N(α + Xβ, σ²) on standardised y. Heteroscedastic BLR
         (``alpha_v_samples`` + ``beta_v_samples``) is handled via the same
         branch in :func:`posterior_predictive_coverage`; the density figure uses
         the homoscedastic-style σ above.
      3. Otherwise skipped with a printed note (e.g. LogNormal / BNN / SAR
         before they register pp_samples).
    """
    import matplotlib.pyplot as plt
    s = slice_step_fn(target_step)
    y_true = s["y_test"] * s["y_std"] + s["y_mean"]
    X_test = s["X_test"]
    y_mean_g, y_std_g = s["y_mean"], s["y_std"]
    n_expected = 1 + X_test.shape[1]

    def _eligible(name):
        if target_step not in models.get(name, {}):
            return False
        p = models[name][target_step]
        if p.get("pp_samples") is not None:
            return True
        c, sgm = p.get("coefs_samples"), p.get("sigma_samples")
        return (c is not None and c.shape[0] > 1
                 and c.shape[-1] == n_expected
                 and sgm is not None)

    eligible = [m for m in models if _eligible(m)]
    skipped  = [m for m in models if target_step in models[m] and not _eligible(m)]
    if skipped:
        print(f"posterior_predictive_density_figure: skipping (no pp_samples and "
              f"not identity-link Gaussian BLR): {skipped}")
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
        payload = models[name][target_step]
        if payload.get("pp_samples") is not None:
            pp_arr = np.asarray(payload["pp_samples"])
            n_use = min(n_pp, pp_arr.shape[0])
            draw_idx = rng.choice(pp_arr.shape[0], size=n_use, replace=False)
            pp = pp_arr[draw_idx].ravel()
        else:
            coefs = payload["coefs_samples"]
            sigma = payload["sigma_samples"]
            n_use = min(n_pp, coefs.shape[0])
            draw_idx = rng.choice(coefs.shape[0], size=n_use, replace=False)
            mean_std  = coefs[draw_idx, 0:1].T + X_test @ coefs[draw_idx, 1:].T
            sigma_std = sigma[draw_idx][None, :]
            pp_std    = mean_std + sigma_std * rng.normal(size=mean_std.shape)
            pp = (pp_std * y_std_g + y_mean_g).ravel()
        ax.hist(y_true, bins=80, density=True, alpha=0.55, label="observed y",            color="C0")
        ax.hist(pp,     bins=80, density=True, alpha=0.55, label="posterior predictive",  color="C3")
        ax.set_title(name); ax.set_xlabel("ΔV (m³/ha)")
        ax.legend(fontsize=8)
    for ax in axes[len(eligible):]:
        ax.set_visible(False)
    fig.suptitle(f"Posterior-predictive vs observed y (n_cells = {target_step!r})", fontsize=11)
    plt.show()
