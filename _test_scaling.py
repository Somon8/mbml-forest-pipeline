"""Smoke test: scaling sweep at n_cells={25,100} with reduced step counts.
Verifies orchestration works for the full notebook flow."""
import sys, warnings, time
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, ".")
import notebook_helpers as H
from notebook_helpers import SEED, FEATURE_COLS

df_raw = H.load_dataset("out_10km_idx_preprocessed.csv")
df = H.apply_filter_and_features(df_raw)
splits = H.load_splits()

df_train_full = df[df["BK"].isin(set(splits.train_bk_ordered))]
X_full, y_full, g_full, _ = H.matrices(df_train_full)
mu_x, sigma_x = H.fit_scaler(X_full)

KWARGS = {
    "linear": {},
    "exact_gp": {"n_steps": 50},
    "svgp": {"n_inducing": 128, "n_steps": 100, "batch_size": 4096},
    "rf": {"n_estimators": 30},
}
EXACT_GP_MAX_BK = 25

for n_cells in [25, 100]:
    df_train, df_test = H.split_frames(df, splits, n_cells)
    X_tr_raw, y_tr, g_tr, _ = H.matrices(df_train)
    X_te_raw, y_te, g_te, coords_te = H.matrices(df_test)
    X_tr = H.apply_scaler(X_tr_raw, mu_x, sigma_x)
    X_te = H.apply_scaler(X_te_raw, mu_x, sigma_x)
    n_bk = len(H.get_train_bk(splits, n_cells))
    print(f"\n=== n_cells={n_cells} ({n_bk} BK, {len(X_tr):,} pixels) ===",
          flush=True)
    label = f"group_n{n_cells}"
    for name in ["linear", "rf", "svgp", "exact_gp"]:
        if name == "exact_gp" and int(n_cells) > EXACT_GP_MAX_BK:
            print(f"  skip exact_gp@{n_cells}: too large", flush=True)
            continue
        H.run_one(name, label, X_tr, y_tr, g_tr, X_te, y_te, g_te, coords_te,
                  n_cells_eff=n_cells, n_train_bk=n_bk,
                  model_kwargs=KWARGS[name])

print("\n--- cache-hit re-run for n_cells=25 ---", flush=True)
t0 = time.time()
df_train, df_test = H.split_frames(df, splits, 25)
X_tr_raw, y_tr, g_tr, _ = H.matrices(df_train)
X_te_raw, y_te, g_te, coords_te = H.matrices(df_test)
X_tr = H.apply_scaler(X_tr_raw, mu_x, sigma_x)
X_te = H.apply_scaler(X_te_raw, mu_x, sigma_x)
for name in ["linear", "rf", "svgp"]:
    H.run_one(name, "group_n25", X_tr, y_tr, g_tr, X_te, y_te, g_te, coords_te,
              n_cells_eff=25, n_train_bk=25, model_kwargs=KWARGS[name])
print(f"(cached re-run for 3 models: {time.time() - t0:.2f}s)", flush=True)

results = H.aggregate_results()
print(f"\nAggregate: {len(results)} cached records.")
print(results[["model", "n_cells_eff", "rmse", "fit_seconds"]].to_string(index=False))
