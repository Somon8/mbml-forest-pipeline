"""Exact Gaussian Process regression in Pyro (Matérn-5/2 ARD kernel).

Only feasible at small N (≤ ~10k) — the kernel matrix is N×N. The
notebook restricts this model to ``n_cells ≤ 25`` and skips it
elsewhere with a logged reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyro
import pyro.contrib.gp as gp
import torch
from pyro.contrib.gp.kernels import Matern52

from .base import BaseModel

SEED = 42
DEVICE = torch.device("cpu")
DEFAULT_DTYPE = torch.float32


class ExactGPModel(BaseModel):
    name = "exact_gp"
    is_probabilistic = True

    def __init__(self, n_steps: int = 500, lr: float = 0.01, jitter: float = 1e-4):
        self.n_steps = n_steps
        self.lr = lr
        self.jitter = jitter
        self.gp_model: gp.models.GPRegression | None = None
        self._n_features: int | None = None

    def _to_tensor(self, X):
        return torch.as_tensor(X, dtype=DEFAULT_DTYPE, device=DEVICE)

    def fit(self, X_train, y_train, *, group_train=None):
        def _do():
            pyro.clear_param_store()
            pyro.set_rng_seed(SEED)
            torch.manual_seed(SEED)
            X = self._to_tensor(X_train)
            y = self._to_tensor(y_train)
            self._n_features = X.shape[1]
            kernel = Matern52(
                input_dim=self._n_features,
                variance=torch.tensor(1.0),
                lengthscale=torch.ones(self._n_features),
            )
            self.gp_model = gp.models.GPRegression(
                X, y, kernel, noise=torch.tensor(1.0), jitter=self.jitter,
            )
            optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=self.lr)
            loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
            losses = []
            for _ in range(self.n_steps):
                optimizer.zero_grad()
                loss = loss_fn(self.gp_model.model, self.gp_model.guide)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            return {"final_elbo_loss": float(losses[-1])}

        return self._timed_fit(_do)

    def predict(self, X_test, *, n_samples=200):
        assert self.gp_model is not None
        X = self._to_tensor(X_test)
        with torch.no_grad():
            mean, cov = self.gp_model(X, full_cov=False, noiseless=False)
        m = mean.detach().cpu().numpy()
        v = cov.detach().cpu().numpy()
        std = np.sqrt(np.maximum(v, 1e-12))
        return m, std

    def predict_samples(self, X_test, n_samples=200):
        m, std = self.predict(X_test)
        rng = np.random.default_rng(SEED)
        return rng.normal(loc=m, scale=std, size=(n_samples, len(m)))

    def save(self, path: Path):
        state = {
            "param_store": pyro.get_param_store().get_state(),
            "n_features": self._n_features,
            "n_steps": self.n_steps,
            "lr": self.lr,
            "jitter": self.jitter,
        }
        if self.gp_model is not None:
            state["X_train"] = self.gp_model.X.detach().cpu().numpy()
            state["y_train"] = self.gp_model.y.detach().cpu().numpy()
        torch.save(state, Path(path))

    def load(self, path: Path):
        state = torch.load(Path(path), weights_only=False)
        pyro.clear_param_store()
        self._n_features = state["n_features"]
        self.n_steps = state["n_steps"]
        self.lr = state["lr"]
        self.jitter = state["jitter"]
        X = self._to_tensor(state["X_train"])
        y = self._to_tensor(state["y_train"])
        kernel = Matern52(
            input_dim=self._n_features,
            variance=torch.tensor(1.0),
            lengthscale=torch.ones(self._n_features),
        )
        self.gp_model = gp.models.GPRegression(
            X, y, kernel, noise=torch.tensor(1.0), jitter=self.jitter,
        )
        pyro.get_param_store().set_state(state["param_store"])
