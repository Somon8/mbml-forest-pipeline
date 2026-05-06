"""Shared helpers for ``04_model_comparison.ipynb``.

Putting the heavy logic here keeps notebook cells short and means the
notebook can re-run end to end on a warm cache by orchestrating already
cached pieces.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm as _norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models import MODELS


SEED = 42
PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "cache"
CACHE_RESULTS = CACHE_DIR / "results"
CACHE_MODELS = CACHE_DIR / "models"
SPLITS_JSON = CACHE_DIR / "splits.json"
DATA_PARQUET = CACHE_DIR / "data.parquet"

# Z-value for a central 90 % normal interval.
Z90 = 1.6448536269514722


# ----------------------------------------------------------------------
# Feature config
# ----------------------------------------------------------------------

# The feature list is leakage-aware: only omdrev1 / time-invariant /
# log-transformed flow + species-composition columns. See PROJECT_JOURNAL.md
# §2.7 (log1p flow) and the task brief for the full justification.
FEATURE_COLS = [
    "volym_omdrev1",
    "biomassa_omdrev1",
    "grundyta_omdrev1",
    "medelhojd_omdrev1",
    "medeldiameter_omdrev1",
    "p95_omdrev1",
    "vegetationskvot_omdrev1",
    "slu_skogskarta_gran_volym",
    "slu_skogskarta_tall_volym",
    "slu_skogskarta_bjork_volym",
    "markfuktighet",
    "log1p_flodesackumulering",
    "x",
    "y",
]

FORBIDDEN_FEATURES = {
    # any *_omdrev2 column
    "tradhojd",
    "dist_to_no_forest_m",
    "dist_to_lake_m",
}

TARGET_COL = "delta_volym"


# ----------------------------------------------------------------------
# Data loading and split construction
# ----------------------------------------------------------------------

def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def apply_filter_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ``is_stable_forest`` and add the engineered columns we need."""

    # Sanity: rebuild is_stable_forest from primitives, just to be defensive.
    primitives = [
        "is_no_forest", "is_lake",
        "delta_neg_medelhojd", "delta_neg_p95", "delta_neg_medeldiameter",
        "delta_neg_biomassa", "delta_neg_volym",
    ]
    for c in primitives:
        if c not in df.columns:
            raise ValueError(f"missing preprocessing column: {c}")
    rebuilt = (
        ~df["is_no_forest"].astype(bool)
        & ~df["is_lake"].astype(bool)
        & ~(df["delta_neg_medelhojd"].astype(bool)
            | df["delta_neg_p95"].astype(bool)
            | df["delta_neg_medeldiameter"].astype(bool)
            | df["delta_neg_biomassa"].astype(bool)
            | df["delta_neg_volym"].astype(bool))
    )
    if "is_stable_forest" in df.columns:
        cur = df["is_stable_forest"].astype(bool)
        if not (rebuilt == cur).all():
            df = df.copy()
            df["is_stable_forest"] = rebuilt
    else:
        df["is_stable_forest"] = rebuilt

    n0 = len(df)
    df = df.loc[df["is_stable_forest"]].copy()
    n1 = len(df)

    df[TARGET_COL] = df["volym_omdrev2"] - df["volym_omdrev1"]
    df["log1p_flodesackumulering"] = np.log1p(df["flodesackumulering"])

    print(f"Rows before mask: {n0:,}")
    print(f"Rows after  mask: {n1:,}")

    bk_counts = df["BK"].value_counts()
    print(f"BK cells with ≥1 surviving pixel: {len(bk_counts):,}")
    print(f"  median pixels per BK: {bk_counts.median():.0f}")
    print(f"  min/max pixels per BK: {bk_counts.min()} / {bk_counts.max()}")

    return df


def assert_no_forbidden(feature_cols):
    bad = []
    for c in feature_cols:
        if c in FORBIDDEN_FEATURES:
            bad.append(c)
        if c.endswith("_omdrev2") or "_omdrev2_" in c:
            bad.append(c)
    if bad:
        raise AssertionError(f"forbidden leaky features: {bad}")


@dataclass
class Splits:
    test_bk: list[str]
    train_bk_ordered: list[str]   # nested order (deterministic shuffle)
    n_cells_grid: list[int | str]
    seed: int = SEED

    def to_dict(self):
        return {
            "test_bk": list(self.test_bk),
            "train_bk_ordered": list(self.train_bk_ordered),
            "n_cells_grid": list(self.n_cells_grid),
            "seed": self.seed,
        }


def build_splits(df: pd.DataFrame, n_cells_grid=(5, 25, 100, 250, "all"),
                 test_frac: float = 0.20, seed: int = SEED) -> Splits:
    """Hold out 20 % of BK cells as test, deterministically order the rest
    so that smaller scaling subsets are nested inside larger ones."""
    bk_all = sorted(df["BK"].unique().tolist())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(bk_all))
    n_test = int(round(test_frac * len(bk_all)))
    test_bk = sorted([bk_all[i] for i in perm[:n_test]])
    train_pool = [bk_all[i] for i in perm[n_test:]]
    # Second deterministic shuffle of the remaining training pool — fixed
    # so that {first 5, first 25, ...} are nested.
    rng2 = np.random.default_rng(seed + 1)
    train_perm = rng2.permutation(len(train_pool))
    train_bk_ordered = [train_pool[i] for i in train_perm]
    return Splits(
        test_bk=test_bk,
        train_bk_ordered=train_bk_ordered,
        n_cells_grid=list(n_cells_grid),
        seed=seed,
    )


def save_splits(splits: Splits, path: Path = SPLITS_JSON):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits.to_dict(), indent=2))


def load_splits(path: Path = SPLITS_JSON) -> Splits:
    d = json.loads(path.read_text())
    return Splits(**d)


def get_train_bk(splits: Splits, n_cells) -> list[str]:
    if n_cells == "all":
        return list(splits.train_bk_ordered)
    return list(splits.train_bk_ordered[: int(n_cells)])


# ----------------------------------------------------------------------
# Feature matrix construction
# ----------------------------------------------------------------------

def split_frames(df: pd.DataFrame, splits: Splits, n_cells):
    train_bk = set(get_train_bk(splits, n_cells))
    test_bk = set(splits.test_bk)
    df_train = df[df["BK"].isin(train_bk)].copy()
    df_test = df[df["BK"].isin(test_bk)].copy()
    return df_train, df_test


def matrices(df_part: pd.DataFrame, feature_cols=FEATURE_COLS):
    X = df_part[feature_cols].to_numpy(dtype=np.float32)
    y = df_part[TARGET_COL].to_numpy(dtype=np.float32)
    g = df_part["BK"].to_numpy()
    coords = df_part[["x", "y"]].to_numpy(dtype=np.float64)
    return X, y, g, coords


def fit_scaler(X_train: np.ndarray):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return mu.astype(np.float32), sigma.astype(np.float32)


def apply_scaler(X, mu, sigma):
    return ((X - mu) / sigma).astype(np.float32)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def gaussian_nlpd(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2)
                         + 0.5 * (y - mu) ** 2 / sigma ** 2))


def gaussian_crps(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    pdf = _norm.pdf(z)
    cdf = _norm.cdf(z)
    crps = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def coverage(y, mu, sigma, alpha=0.90):
    z = _norm.ppf(0.5 + alpha / 2.0)
    sigma = np.maximum(sigma, 1e-6)
    inside = (y >= mu - z * sigma) & (y <= mu + z * sigma)
    return float(np.mean(inside))


def calibration_curve(y, mu, sigma, alphas=None):
    if alphas is None:
        alphas = np.arange(0.1, 0.96, 0.05)
    sigma = np.maximum(sigma, 1e-6)
    emp = []
    for a in alphas:
        z = _norm.ppf(0.5 + a / 2.0)
        inside = (y >= mu - z * sigma) & (y <= mu + z * sigma)
        emp.append(float(np.mean(inside)))
    return np.asarray(alphas), np.asarray(emp)


def point_metrics(y, mu):
    rmse = float(np.sqrt(mean_squared_error(y, mu)))
    mae = float(mean_absolute_error(y, mu))
    r2 = float(r2_score(y, mu))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ----------------------------------------------------------------------
# Experiment loop primitive
# ----------------------------------------------------------------------

def cache_path(model_name: str, label: str) -> Path:
    CACHE_RESULTS.mkdir(parents=True, exist_ok=True)
    return CACHE_RESULTS / f"{model_name}__{label}.pkl"


def model_blob_path(model_name: str, label: str) -> Path:
    CACHE_MODELS.mkdir(parents=True, exist_ok=True)
    # Pyro models save as .pt, sklearn as .joblib; both via .save() so the
    # caller picks. We just use .bin as a generic suffix.
    return CACHE_MODELS / f"{model_name}__{label}.bin"


def run_one(
    model_name: str,
    label: str,
    X_train, y_train, g_train,
    X_test, y_test, g_test, coords_test,
    *,
    is_random_split: bool = False,
    n_cells_eff: int | str = None,
    n_train_bk: int | None = None,
    force: bool = False,
    n_pred_samples: int = 200,
    model_kwargs: dict | None = None,
):
    """Fit one model on one training subset, score on the test set, cache.

    The cache key is ``<model>__<label>``. Re-run with ``force=True`` to
    overwrite. Returns the cached dict (predictions + metrics + meta).
    """
    out_path = cache_path(model_name, label)
    if out_path.exists() and not force:
        with out_path.open("rb") as f:
            return pickle.load(f)

    print(f"  fitting {model_name}@{label} (n_train={len(X_train):,}) ...",
          flush=True)
    cls = MODELS[model_name]
    model = cls(**(model_kwargs or {}))
    fit_info = model.fit(
        X_train, y_train,
        group_train=g_train if model_name == "hierarchical" else None,
    )

    t_pred = time.perf_counter()
    if model_name == "hierarchical":
        mu, sigma = model.predict(X_test, n_samples=n_pred_samples,
                                  group_test=g_test)
    else:
        mu, sigma = model.predict(X_test, n_samples=n_pred_samples)
    pred_seconds = time.perf_counter() - t_pred

    metrics = point_metrics(y_test, mu)
    if sigma is not None:
        metrics["nlpd"] = gaussian_nlpd(y_test, mu, sigma)
        metrics["crps"] = gaussian_crps(y_test, mu, sigma)
        metrics["coverage_90"] = coverage(y_test, mu, sigma, alpha=0.90)
        alphas, emp = calibration_curve(y_test, mu, sigma)
    else:
        metrics["nlpd"] = np.nan
        metrics["crps"] = np.nan
        metrics["coverage_90"] = np.nan
        alphas, emp = None, None

    blob_path = model_blob_path(model_name, label)
    try:
        model.save(blob_path)
    except Exception as e:
        print(f"    note: {model_name} save failed ({e}); skipping artifact")
        blob_path = None

    record = {
        "model": model_name,
        "label": label,
        "is_random_split": bool(is_random_split),
        "n_cells_eff": n_cells_eff,
        "n_train_bk": n_train_bk,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "fit_seconds": float(fit_info.get("fit_seconds", np.nan)),
        "predict_seconds": float(pred_seconds),
        "peak_memory_mb": float(fit_info.get("peak_memory_mb", np.nan)),
        "delta_memory_mb": float(fit_info.get("delta_memory_mb", np.nan)),
        "metrics": metrics,
        "y_test": np.asarray(y_test, dtype=np.float32),
        "mu": np.asarray(mu, dtype=np.float32),
        "sigma": (None if sigma is None else np.asarray(sigma, dtype=np.float32)),
        "coords_test": np.asarray(coords_test, dtype=np.float64),
        "alphas": alphas,
        "emp_coverage": emp,
        "model_artifact": str(blob_path) if blob_path else None,
    }
    with out_path.open("wb") as f:
        pickle.dump(record, f)
    print(f"    rmse={metrics['rmse']:6.2f}  fit={record['fit_seconds']:.1f}s"
          f"  peak_rss={record['peak_memory_mb']:.0f}MB", flush=True)
    return record


def aggregate_results():
    """Concat all cached result pickles into one DataFrame (no predictions)."""
    rows = []
    for p in sorted(CACHE_RESULTS.glob("*.pkl")):
        with p.open("rb") as f:
            rec = pickle.load(f)
        flat = {
            "model": rec["model"],
            "label": rec["label"],
            "is_random_split": rec["is_random_split"],
            "n_cells_eff": rec["n_cells_eff"],
            "n_train_bk": rec["n_train_bk"],
            "n_train": rec["n_train"],
            "n_test": rec["n_test"],
            "fit_seconds": rec["fit_seconds"],
            "predict_seconds": rec["predict_seconds"],
            "peak_memory_mb": rec["peak_memory_mb"],
            "delta_memory_mb": rec["delta_memory_mb"],
        }
        for k, v in rec["metrics"].items():
            flat[k] = v
        rows.append(flat)
    return pd.DataFrame(rows)


def load_record(model_name: str, label: str) -> dict:
    p = cache_path(model_name, label)
    with p.open("rb") as f:
        return pickle.load(f)
