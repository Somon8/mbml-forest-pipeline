"""Bayesian Neural Network: 2 hidden layers of 64 ReLU units, mean-field
Normal weight priors, learned output noise.

Trained with SVI (AutoNormal) and Adam. Minibatched at 4096 rows once
the training set is large enough that full-batch is wasteful.
"""

from __future__ import annotations

import math
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

HIDDEN = 64
N_HIDDEN_LAYERS = 2
PRIOR_W_SCALE = 1.0


def _bnn_forward(X, params):
    h = X
    for i, (W, b) in enumerate(params[:-1]):
        h = torch.relu(h @ W + b)
    W_out, b_out = params[-1]
    return (h @ W_out + b_out).squeeze(-1)


def bnn_model(X, y=None):
    n, d = X.shape
    sizes = [d] + [HIDDEN] * N_HIDDEN_LAYERS + [1]
    params = []
    for i in range(len(sizes) - 1):
        in_, out = sizes[i], sizes[i + 1]
        scale = PRIOR_W_SCALE / math.sqrt(in_)
        W = pyro.sample(
            f"W{i}",
            dist.Normal(torch.zeros(in_, out), scale * torch.ones(in_, out)).to_event(2),
        )
        b = pyro.sample(
            f"b{i}",
            dist.Normal(torch.zeros(out), torch.ones(out)).to_event(1),
        )
        params.append((W, b))
    sigma = pyro.sample("sigma", dist.HalfNormal(torch.tensor(50.0)))
    mean = _bnn_forward(X, params)
    with pyro.plate("data", n):
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=y)


class BNNModel(BaseModel):
    name = "bnn"
    is_probabilistic = True

    def __init__(self, n_steps: int = 4000, lr: float = 0.005,
                 batch_size: int = 4096):
        self.n_steps = n_steps
        self.lr = lr
        self.batch_size = batch_size
        self.guide: AutoNormal | None = None
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
            n = X.shape[0]
            self._n_features = X.shape[1]
            self.guide = AutoNormal(bnn_model)
            svi = SVI(bnn_model, self.guide, Adam({"lr": self.lr}), Trace_ELBO())

            rng = np.random.default_rng(SEED)
            full_batch = n <= self.batch_size
            losses = []
            for step in range(self.n_steps):
                if full_batch:
                    loss = svi.step(X, y)
                else:
                    idx = rng.integers(0, n, size=self.batch_size)
                    loss = svi.step(X[idx], y[idx])
                losses.append(loss)
            return {"final_elbo_loss": float(losses[-1])}

        return self._timed_fit(_do)

    def _predictive(self, X_test, n_samples):
        """Sample (S, n_test) draws by tracing the guide and reconstructing
        the forward pass per draw. Bypasses ``Predictive`` to avoid its
        parallel-mode batch dim issues with ``X @ W``."""
        assert self.guide is not None
        X = self._to_tensor(X_test)
        dummy = torch.zeros(1, self._n_features, dtype=DEFAULT_DTYPE)
        sizes = [self._n_features] + [HIDDEN] * N_HIDDEN_LAYERS + [1]
        n_layers = len(sizes) - 1
        out_samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                tr = poutine.trace(self.guide).get_trace(dummy)
                params = []
                for i in range(n_layers):
                    W = tr.nodes[f"W{i}"]["value"].detach()
                    b = tr.nodes[f"b{i}"]["value"].detach()
                    params.append((W, b))
                sigma = tr.nodes["sigma"]["value"].detach()
                mean = _bnn_forward(X, params)
                noise = torch.randn(X.shape[0]) * sigma
                out_samples.append((mean + noise).cpu().numpy())
        return np.stack(out_samples, axis=0)

    def predict(self, X_test, *, n_samples=200):
        obs = self._predictive(X_test, n_samples)
        return obs.mean(axis=0), obs.std(axis=0, ddof=1)

    def predict_samples(self, X_test, n_samples=200):
        return self._predictive(X_test, n_samples)

    def save(self, path: Path):
        state = {
            "param_store": pyro.get_param_store().get_state(),
            "n_features": self._n_features,
            "n_steps": self.n_steps,
            "lr": self.lr,
            "batch_size": self.batch_size,
        }
        torch.save(state, Path(path))

    def load(self, path: Path):
        state = torch.load(Path(path), weights_only=False)
        pyro.clear_param_store()
        pyro.get_param_store().set_state(state["param_store"])
        self._n_features = state["n_features"]
        self.n_steps = state["n_steps"]
        self.lr = state["lr"]
        self.batch_size = state["batch_size"]
        self.guide = AutoNormal(bnn_model)
        dummy_X = torch.zeros(1, self._n_features, dtype=DEFAULT_DTYPE)
        self.guide(dummy_X)
