"""Hierarchical Bayesian regression with partial pooling on BK indexruta.

Model:
    Hyperprior on group intercept distribution:
        mu_alpha   ~ Normal(0, 50)        # global growth rate (m^3/ha)
        sigma_alpha ~ HalfNormal(50)
    Group intercepts:
        alpha_g    ~ Normal(mu_alpha, sigma_alpha) for each BK
    Fixed slopes:
        w          ~ Normal(0, 1)
    Likelihood:
        sigma      ~ HalfNormal(50)
        y_n        ~ Normal(alpha[g_n] + w @ x_n, sigma)

The notebook supplies dense integer codes (0..G-1) per training row via
``group_train``; unseen test groups fall back to mu_alpha.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam

from .base import BaseModel

SEED = 42
DEVICE = torch.device("cpu")
DEFAULT_DTYPE = torch.float32


def hier_model(X, group_idx, n_groups, y=None):
    n, d = X.shape
    mu_alpha = pyro.sample("mu_alpha", dist.Normal(torch.tensor(0.0), torch.tensor(50.0)))
    sigma_alpha = pyro.sample("sigma_alpha", dist.HalfNormal(torch.tensor(50.0)))
    with pyro.plate("groups", n_groups):
        alpha = pyro.sample("alpha", dist.Normal(mu_alpha, sigma_alpha))
    w = pyro.sample("w", dist.Normal(torch.zeros(d), torch.ones(d)).to_event(1))
    sigma = pyro.sample("sigma", dist.HalfNormal(torch.tensor(50.0)))
    mean = alpha[group_idx] + X @ w
    with pyro.plate("data", n):
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=y)


class HierarchicalModel(BaseModel):
    name = "hierarchical"
    is_probabilistic = True

    def __init__(self, n_steps: int = 2500, lr: float = 0.01):
        self.n_steps = n_steps
        self.lr = lr
        self.guide: AutoNormal | None = None
        self._group_to_idx: dict | None = None
        self._n_groups: int | None = None
        self._n_features: int | None = None

    def _to_tensor(self, X):
        return torch.as_tensor(X, dtype=DEFAULT_DTYPE, device=DEVICE)

    def _encode_groups(self, group_train, fit=True):
        if fit:
            uniq = np.unique(group_train)
            self._group_to_idx = {g: i for i, g in enumerate(uniq.tolist())}
            self._n_groups = len(uniq)
        codes = np.array(
            [self._group_to_idx[g] if g in self._group_to_idx else -1
             for g in group_train],
            dtype=np.int64,
        )
        return codes

    def fit(self, X_train, y_train, *, group_train=None):
        if group_train is None:
            raise ValueError("HierarchicalModel.fit requires group_train")

        def _do():
            pyro.clear_param_store()
            pyro.set_rng_seed(SEED)
            X = self._to_tensor(X_train)
            y = self._to_tensor(y_train)
            codes = self._encode_groups(np.asarray(group_train), fit=True)
            g = torch.as_tensor(codes, dtype=torch.long, device=DEVICE)
            self._n_features = X.shape[1]
            self.guide = AutoNormal(hier_model)
            svi = SVI(hier_model, self.guide, Adam({"lr": self.lr}), Trace_ELBO())
            losses = []
            for _ in range(self.n_steps):
                loss = svi.step(X, g, self._n_groups, y)
                losses.append(loss)
            return {"final_elbo_loss": float(losses[-1]),
                    "n_groups_train": int(self._n_groups)}

        return self._timed_fit(_do)

    def predict(self, X_test, *, n_samples=200, group_test=None):
        return self._predict(X_test, n_samples, group_test, want_samples=False)

    def predict_samples(self, X_test, n_samples=200, group_test=None):
        return self._predict(X_test, n_samples, group_test, want_samples=True)

    def _predict(self, X_test, n_samples, group_test, want_samples):
        assert self.guide is not None
        X = self._to_tensor(X_test)
        if group_test is None:
            codes = np.full(len(X_test), -1, dtype=np.int64)
        else:
            codes = self._encode_groups(np.asarray(group_test), fit=False)

        # Sample latents directly from the guide. Bypassing Predictive avoids
        # the parallel-mode batch dim that breaks ``alpha[group_idx] + X @ w``
        # when w gets an extra leading sample axis.
        dummy_X = torch.zeros(1, self._n_features, dtype=DEFAULT_DTYPE)
        dummy_g = torch.zeros(1, dtype=torch.long)
        sites = ["w", "sigma", "mu_alpha", "sigma_alpha", "alpha"]
        bag = {k: [] for k in sites}
        with torch.no_grad():
            for _ in range(n_samples):
                tr = poutine.trace(self.guide).get_trace(
                    dummy_X, dummy_g, self._n_groups,
                )
                for k in sites:
                    bag[k].append(tr.nodes[k]["value"].detach())
        w = torch.stack(bag["w"])                    # (S, d)
        sigma = torch.stack(bag["sigma"])            # (S,)
        mu_alpha = torch.stack(bag["mu_alpha"])      # (S,)
        sigma_alpha = torch.stack(bag["sigma_alpha"])  # (S,)
        alpha_train = torch.stack(bag["alpha"])      # (S, G)

        S = w.shape[0]
        # Build a full alpha matrix with unseen groups drawn from the
        # hyperprior so unseen-group predictions still reflect uncertainty.
        unseen_mask = codes < 0
        # Indices we will read from alpha_train must be valid; for unseen
        # rows we read from a parallel "fresh draw" tensor.
        safe_codes = np.where(unseen_mask, 0, codes)
        idx = torch.as_tensor(safe_codes, dtype=torch.long)

        seen_alpha = alpha_train[:, idx]   # (S, n_test)
        eps = torch.randn(S, len(X_test))
        unseen_alpha = mu_alpha[:, None] + sigma_alpha[:, None] * eps
        unseen_t = torch.as_tensor(unseen_mask, dtype=torch.bool)
        alpha_full = torch.where(unseen_t.unsqueeze(0), unseen_alpha, seen_alpha)

        mean = alpha_full + (w @ X.T)      # (S, n_test)
        # Add observation noise to get predictive draws.
        noise = torch.randn(S, len(X_test)) * sigma[:, None]
        obs = (mean + noise).cpu().numpy()
        if want_samples:
            return obs
        return obs.mean(axis=0), obs.std(axis=0, ddof=1)

    def save(self, path: Path):
        state = {
            "param_store": pyro.get_param_store().get_state(),
            "group_to_idx": self._group_to_idx,
            "n_groups": self._n_groups,
            "n_features": self._n_features,
            "n_steps": self.n_steps,
            "lr": self.lr,
        }
        torch.save(state, Path(path))

    def load(self, path: Path):
        state = torch.load(Path(path), weights_only=False)
        pyro.clear_param_store()
        pyro.get_param_store().set_state(state["param_store"])
        self._group_to_idx = state["group_to_idx"]
        self._n_groups = state["n_groups"]
        self._n_features = state["n_features"]
        self.n_steps = state["n_steps"]
        self.lr = state["lr"]
        self.guide = AutoNormal(hier_model)
        dummy_X = torch.zeros(1, self._n_features, dtype=DEFAULT_DTYPE)
        dummy_g = torch.zeros(1, dtype=torch.long)
        self.guide(dummy_X, dummy_g, self._n_groups)
