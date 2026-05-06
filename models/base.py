"""BaseModel ABC plus shared resource-tracking helpers.

Every model in this package fits, predicts, and serialises through the
same surface so the experiment loop can treat them as interchangeable.
``fit`` always returns a dict with ``fit_seconds`` and ``peak_memory_mb``.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import psutil


class _PeakRSSSampler:
    """Sample the current process's RSS in a background thread.

    psutil RSS is preferred over tracemalloc for these workloads because
    Pyro's autograd graphs and PyTorch tensors live outside Python's
    allocator — tracemalloc misses them. RSS is sampled every 50 ms.
    """

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._peak_bytes = self._proc.memory_info().rss
        self._baseline = self._peak_bytes
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._stop.clear()
        self._baseline = self._proc.memory_info().rss
        self._peak_bytes = self._baseline
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return False

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
            except psutil.Error:
                return
            if rss > self._peak_bytes:
                self._peak_bytes = rss
            self._stop.wait(self.interval)

    @property
    def peak_mb(self) -> float:
        return self._peak_bytes / (1024 * 1024)

    @property
    def delta_mb(self) -> float:
        return max(0.0, (self._peak_bytes - self._baseline) / (1024 * 1024))


class BaseModel(ABC):
    """Common API for every model in the comparison."""

    name: str = "base"
    is_probabilistic: bool = False

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        group_train: np.ndarray | None = None,
    ) -> dict:
        """Fit the model. Must return at least ``fit_seconds`` and
        ``peak_memory_mb``. ``group_train`` carries integer BK ids per row
        and is used only by the hierarchical model."""

    @abstractmethod
    def predict(
        self,
        X_test: np.ndarray,
        *,
        n_samples: int = 200,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Returns (mean, std). ``std`` is ``None`` for non-probabilistic
        models. ``mean`` always has shape (n_test,)."""

    def predict_samples(
        self, X_test: np.ndarray, n_samples: int = 200
    ) -> np.ndarray | None:
        """Returns (n_samples, n_test) draws from the posterior predictive,
        or ``None`` when not available."""
        return None

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...

    def _timed_fit(self, fit_inner) -> dict:
        """Helper: wrap a closure that does the actual fit, returning
        timing/memory metadata. Subclasses call this from inside ``fit``."""
        with _PeakRSSSampler() as sampler:
            t0 = time.perf_counter()
            extra = fit_inner() or {}
            elapsed = time.perf_counter() - t0
        info = {
            "fit_seconds": float(elapsed),
            "peak_memory_mb": float(sampler.peak_mb),
            "delta_memory_mb": float(sampler.delta_mb),
        }
        info.update(extra)
        return info
