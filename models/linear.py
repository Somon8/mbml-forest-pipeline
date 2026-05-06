"""Frequentist linear regression baseline (sklearn).

Speed/sanity reference. Returns no predictive uncertainty.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from .base import BaseModel

SEED = 42


class LinearModel(BaseModel):
    name = "linear"
    is_probabilistic = False

    def __init__(self):
        self.model: LinearRegression | None = None

    def fit(self, X_train, y_train, *, group_train=None):
        def _do():
            self.model = LinearRegression()
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
