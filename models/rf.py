"""Random Forest regressor with empirical leaf-prediction std as a
crude predictive uncertainty.

The std comes from the variation across trees, not a posterior — but
it is a useful baseline against which to compare the Bayesian models'
calibration. Treat it as a frequentist epistemic estimate.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import BaseModel

SEED = 42


class RFModel(BaseModel):
    name = "rf"
    is_probabilistic = True

    def __init__(self, n_estimators: int = 200, max_depth: int | None = None,
                 n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.model: RandomForestRegressor | None = None

    def fit(self, X_train, y_train, *, group_train=None):
        def _do():
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=SEED,
                n_jobs=self.n_jobs,
            )
            self.model.fit(X_train, y_train)

        return self._timed_fit(_do)

    def _per_tree_predictions(self, X):
        return np.stack(
            [tree.predict(X) for tree in self.model.estimators_], axis=0
        )

    def predict(self, X_test, *, n_samples=200):
        assert self.model is not None
        per_tree = self._per_tree_predictions(X_test)  # (T, n_test)
        mean = per_tree.mean(axis=0)
        std = per_tree.std(axis=0, ddof=1)
        std = np.maximum(std, 1e-6)
        return mean, std

    def predict_samples(self, X_test, n_samples=200):
        per_tree = self._per_tree_predictions(X_test)
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, per_tree.shape[0], size=n_samples)
        return per_tree[idx]

    def save(self, path: Path):
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)
