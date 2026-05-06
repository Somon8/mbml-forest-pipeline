"""Sparse Variational GP (Pyro ``VariationalSparseGP``) with Matérn-5/2
ARD kernel and Gaussian likelihood.

Inducing points (~512) are initialised by k-means on the training inputs.
Trained with Adam + minibatching at batch_size=4096.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyro
import pyro.contrib.gp as gp
import torch
from pyro.contrib.gp.kernels import Matern52
from pyro.contrib.gp.likelihoods import Gaussian
from sklearn.cluster import MiniBatchKMeans

from .base import BaseModel

SEED = 42
DEVICE = torch.device("cpu")
DEFAULT_DTYPE = torch.float32


class SVGPModel(BaseModel):
    name = "svgp"
    is_probabilistic = True

    def __init__(self, n_inducing: int = 512, n_steps: int = 2000,
                 lr: float = 0.01, batch_size: int = 4096,
                 jitter: float = 1e-4):
        self.n_inducing = n_inducing
        self.n_steps = n_steps
        self.lr = lr
        self.batch_size = batch_size
        self.jitter = jitter
        self.gp_model: gp.models.VariationalSparseGP | None = None
        self._n_features: int | None = None

    def _to_tensor(self, X):
        return torch.as_tensor(X, dtype=DEFAULT_DTYPE, device=DEVICE)

    def _init_inducing(self, X_train):
        n_inducing = min(self.n_inducing, X_train.shape[0])
        if n_inducing >= X_train.shape[0]:
            return X_train.copy()
        km = MiniBatchKMeans(
            n_clusters=n_inducing, random_state=SEED,
            batch_size=4096, n_init=3, max_iter=50,
        )
        km.fit(X_train)
        return km.cluster_centers_.astype(np.float32)

    def fit(self, X_train, y_train, *, group_train=None):
        def _do():
            pyro.clear_param_store()
            pyro.set_rng_seed(SEED)
            torch.manual_seed(SEED)
            X = self._to_tensor(X_train)
            y = self._to_tensor(y_train)
            n = X.shape[0]
            self._n_features = X.shape[1]

            Xu_np = self._init_inducing(np.asarray(X_train))
            Xu = torch.as_tensor(Xu_np, dtype=DEFAULT_DTYPE, device=DEVICE)

            kernel = Matern52(
                input_dim=self._n_features,
                variance=torch.tensor(1.0),
                lengthscale=torch.ones(self._n_features),
            )
            likelihood = Gaussian(variance=torch.tensor(1.0))
            self.gp_model = gp.models.VariationalSparseGP(
                X, y, kernel, Xu=Xu, likelihood=likelihood,
                num_data=n, whiten=True, jitter=self.jitter,
            )

            optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=self.lr)
            loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

            rng = np.random.default_rng(SEED)
            full_batch = n <= self.batch_size
            losses = []
            for _ in range(self.n_steps):
                if full_batch:
                    Xb, yb = X, y
                else:
                    idx = rng.integers(0, n, size=self.batch_size)
                    Xb, yb = X[idx], y[idx]
                self.gp_model.set_data(Xb, yb)
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
            f_mean, f_var = self.gp_model(X, full_cov=False)
        # Add observation-noise variance from the Gaussian likelihood
        noise_var = self.gp_model.likelihood.variance.detach()
        m = f_mean.detach().cpu().numpy()
        total_var = (f_var.detach() + noise_var).cpu().numpy()
        std = np.sqrt(np.maximum(total_var, 1e-12))
        return m, std

    def predict_samples(self, X_test, n_samples=200):
        m, std = self.predict(X_test)
        rng = np.random.default_rng(SEED)
        return rng.normal(loc=m, scale=std, size=(n_samples, len(m)))

    def save(self, path: Path):
        state = {
            "param_store": pyro.get_param_store().get_state(),
            "n_features": self._n_features,
            "n_inducing": self.n_inducing,
            "n_steps": self.n_steps,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "jitter": self.jitter,
        }
        if self.gp_model is not None:
            state["X_train"] = self.gp_model.X.detach().cpu().numpy()
            state["y_train"] = self.gp_model.y.detach().cpu().numpy()
            state["Xu"] = self.gp_model.Xu.detach().cpu().numpy()
        torch.save(state, Path(path))

    def load(self, path: Path):
        state = torch.load(Path(path), weights_only=False)
        pyro.clear_param_store()
        self._n_features = state["n_features"]
        self.n_inducing = state["n_inducing"]
        self.n_steps = state["n_steps"]
        self.lr = state["lr"]
        self.batch_size = state["batch_size"]
        self.jitter = state["jitter"]
        X = self._to_tensor(state["X_train"])
        y = self._to_tensor(state["y_train"])
        Xu = self._to_tensor(state["Xu"])
        kernel = Matern52(
            input_dim=self._n_features,
            variance=torch.tensor(1.0),
            lengthscale=torch.ones(self._n_features),
        )
        likelihood = Gaussian(variance=torch.tensor(1.0))
        self.gp_model = gp.models.VariationalSparseGP(
            X, y, kernel, Xu=Xu, likelihood=likelihood,
            num_data=X.shape[0], whiten=True, jitter=self.jitter,
        )
        pyro.get_param_store().set_state(state["param_store"])
