"""
point02_permutation.py

Addresses concern about inflated p-values from train-test split.
Shuffles target N=1000 times and shows actual model R squared exceeds
all permuted-null values. Also runs temporal validation.

Output: Permutation test p-value, temporal validation results.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from shared import (
    load_data, apply_full_preprocessing, make_preprocessor, sdca,
    FEATURES, TARGET, RANDOM_STATE
)

N_PERMUTATIONS = 1000


def run():
    print("\n" + "=" * 80)
    print("POINT 2 — PERMUTATION TEST (N=1000) & TEMPORAL VALIDATION")
    print("=" * 80)

    # Load and preprocess data
    _, agg_raw, _ = load_data()
    df = apply_full_preprocessing(agg_raw)
    X = df[FEATURES]
    y = df[TARGET]

    # Actual model performance on 80/20 split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    pipe = Pipeline([('pre', make_preprocessor()), ('model', sdca())])
    pipe.fit(Xtr, ytr)
    actual_r2 = r2_score(yte, pipe.predict(Xte))
    print(f"\nActual SDCA R2 (80/20): {actual_r2:.4f}")

    # Permutation test
    perm_r2 = []
    print(f"Running {N_PERMUTATIONS} permutations...")
    for i in range(N_PERMUTATIONS):
        y_shuf = np.random.permutation(y.values)
        Xtr_p, Xte_p, ytr_p, yte_p = train_test_split(
            X, y_shuf, test_size=0.2, random_state=i
        )
        p = Pipeline([('pre', make_preprocessor()), ('model', sdca())])
        p.fit(Xtr_p, ytr_p)
        perm_r2.append(r2_score(yte_p, p.predict(Xte_p)))
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{N_PERMUTATIONS}")

    perm_r2 = np.array(perm_r2)
    p_empirical = (perm_r2 >= actual_r2).sum() / N_PERMUTATIONS
    print(f"\nPermuted null R2: mean={perm_r2.mean():.4f}, SD={perm_r2.std():.4f}")
    print(f"Empirical p-value: {p_empirical:.4f}  "
          f"({'p < 0.001' if p_empirical < 0.001 else f'p = {p_empirical:.4f}'})")

    # Temporal validation: train 2017-2022, test 2023-2025
    train_mask = df['Year'] <= 2022
    test_mask = (df['Year'] >= 2023) & (df['Year'] <= 2025)
    Xtr_t, Xte_t = X[train_mask], X[test_mask]
    ytr_t, yte_t = y[train_mask], y[test_mask]

    pipe_t = Pipeline([('pre', make_preprocessor()), ('model', sdca())])
    pipe_t.fit(Xtr_t, ytr_t)
    pred_t = pipe_t.predict(Xte_t)
    temp_r2 = r2_score(yte_t, pred_t)
    temp_rmse = np.sqrt(mean_squared_error(yte_t, pred_t))

    n_t = len(yte_t)
    temp_adj = 1 - (1 - temp_r2) * (n_t - 1) / (n_t - len(FEATURES) - 1)

    print(f"\nTemporal validation (train 2017-22, test 2023-25):")
    print(f"  R2 = {temp_r2:.4f}, Adj R2 = {temp_adj:.4f}, RMSE = {temp_rmse:,.2f}")
    print(f"  n_train = {train_mask.sum()}, n_test = {test_mask.sum()}")

    return {
        'actual_r2': actual_r2,
        'perm_r2_mean': perm_r2.mean(),
        'perm_r2_std': perm_r2.std(),
        'p_empirical': p_empirical,
        'temp_r2': temp_r2,
        'temp_rmse': temp_rmse,
    }


if __name__ == "__main__":
    run()
