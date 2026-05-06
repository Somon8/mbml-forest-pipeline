"""Histogram gradient-boosting baseline (sklearn).

Point-only by default. Probabilistic metrics are reported as ``n/a``
in the experiment loop. We do not fit quantile heads — the project
brief explicitly allows skipping when not trivial.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .base import BaseModel

SEED = 42


class GBMModel(BaseModel):
    name = "gbm"
    is_probabilistic = False

    def __init__(self, max_iter: int = 300, max_depth: int | None = None):
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.model: HistGradientBoostingRegressor | None = None

    def fit(self, X_train, y_train, *, group_train=None):
        def _do():
            self.model = HistGradientBoostingRegressor(
                max_iter=self.max_iter,
                max_depth=self.max_depth,
                random_state=SEED,
            )
            self.model.fit(X_train, y_train)

        return self._timed_fit(_do)

    def predict(self, X_test, *, n_samples=200):
        assert self.model is not None
        mean = self.model.predict(X_test).astype(np.float64)
        return mean, None

    def save(self, path: Path):
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)
