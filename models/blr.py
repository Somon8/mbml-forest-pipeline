"""Bayesian Linear Regression in Pyro with mean-field SVI.

Model:
    w ~ Normal(0, 1) for each feature; b ~ Normal(0, 5)
    sigma ~ HalfNormal(50)
    y_n ~ Normal(b + w @ x_n, sigma)

Guide: AutoNormal (mean-field). Adam, fixed seed.
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


def blr_model(X, y=None):
    n, d = X.shape
    w = pyro.sample("w", dist.Normal(torch.zeros(d), torch.ones(d)).to_event(1))
    b = pyro.sample("b", dist.Normal(torch.tensor(0.0), torch.tensor(5.0)))
    sigma = pyro.sample("sigma", dist.HalfNormal(torch.tensor(50.0)))
    mean = b + X @ w
    with pyro.plate("data", n):
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=y)


class BLRModel(BaseModel):
    name = "blr"
    is_probabilistic = True

    def __init__(self, n_steps: int = 2000, lr: float = 0.01):
        self.n_steps = n_steps
        self.lr = lr
        self.guide: AutoNormal | None = None
        self._n_features: int | None = None

    def _to_tensor(self, X):
        return torch.as_tensor(X, dtype=DEFAULT_DTYPE, device=DEVICE)

    def fit(self, X_train, y_train, *, group_train=None):
        def _do():
            pyro.clear_param_store()
            pyro.set_rng_seed(SEED)
            X = self._to_tensor(X_train)
            y = self._to_tensor(y_train)
            self._n_features = X.shape[1]
            self.guide = AutoNormal(blr_model)
            svi = SVI(blr_model, self.guide, Adam({"lr": self.lr}), Trace_ELBO())
            losses = []
            for step in range(self.n_steps):
                loss = svi.step(X, y)
                losses.append(loss)
            return {"final_elbo_loss": float(losses[-1])}

        return self._timed_fit(_do)

    def _draw_samples(self, X_test, n_samples):
        """Sample (S, n_test) posterior-predictive draws by tracing the
        guide directly, then computing the predictive likelihood manually.

        Bypasses ``Predictive`` because its parallel-mode batching adds an
        extra leading sample dim that breaks ``X @ w``.
        """
        assert self.guide is not None
        X = self._to_tensor(X_test)
        dummy = torch.zeros(1, self._n_features, dtype=DEFAULT_DTYPE)
        ws, bs, sigmas = [], [], []
        with torch.no_grad():
            for _ in range(n_samples):
                tr = poutine.trace(self.guide).get_trace(dummy)
                ws.append(tr.nodes["w"]["value"].detach())
                bs.append(tr.nodes["b"]["value"].detach())
                sigmas.append(tr.nodes["sigma"]["value"].detach())
        w = torch.stack(ws)         # (S, d)
        b = torch.stack(bs)         # (S,)
        sigma = torch.stack(sigmas)  # (S,)
        S, _ = w.shape
        mean = b[:, None] + (w @ X.T)               # (S, n_test)
        eps = torch.randn(S, X.shape[0])
        obs = mean + sigma[:, None] * eps
        return obs.cpu().numpy()

    def predict(self, X_test, *, n_samples=200):
        obs = self._draw_samples(X_test, n_samples)
        return obs.mean(axis=0), obs.std(axis=0, ddof=1)

    def predict_samples(self, X_test, n_samples=200):
        return self._draw_samples(X_test, n_samples)

    def save(self, path: Path):
        path = Path(path)
        state = {
            "param_store": pyro.get_param_store().get_state(),
            "n_features": self._n_features,
            "n_steps": self.n_steps,
            "lr": self.lr,
        }
        torch.save(state, path)

    def load(self, path: Path):
        state = torch.load(path, weights_only=False)
        pyro.clear_param_store()
        pyro.get_param_store().set_state(state["param_store"])
        self._n_features = state["n_features"]
        self.n_steps = state["n_steps"]
        self.lr = state["lr"]
        self.guide = AutoNormal(blr_model)
        # AutoNormal needs one trace through model to be initialised:
        dummy_X = torch.zeros(1, self._n_features, dtype=DEFAULT_DTYPE)
        self.guide(dummy_X)
