"""
point07_cv.py
DR. RAGA POINT 7 (CRITICAL): 10-Fold Cross-Validation

Replaces single 80/20 split validation with robust 10-fold CV.
Reports mean and standard deviation across folds. Low SD indicates
stable model performance.

Output: Table 4-CV (10-fold cross-validation).
"""
from shared import load_data, apply_full_preprocessing, evaluate, sdca, FEATURES


def run():
    print("\n" + "=" * 80)
    print("POINT 7 — 10-FOLD CROSS-VALIDATION FOR SDCA (Table 4-CV)")
    print("=" * 80)

    # Load and preprocess
    _, agg_raw, _ = load_data()
    df = apply_full_preprocessing(agg_raw)

    # Run 10-fold CV
    result = evaluate(sdca, df, 'SDCA')

    # Print per-fold results
    print(f"\n{'Fold':<8} {'R2':>10} {'Adj R2':>10} {'RMSE':>15} {'MAE':>15}")
    print("-" * 65)
    for i, (r2, rmse, mae) in enumerate(
        zip(result['fold_r2'], result['fold_rmse'], result['fold_mae']), 1
    ):
        n, p = len(df), len(FEATURES)
        adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        print(f"Fold {i:<3} {r2:>10.4f} {adj:>10.4f} {rmse:>15,.2f} {mae:>15,.2f}")

    print("-" * 65)
    print(f"{'Mean':<8} {result['r2_mean']:>10.4f} {result['adj_r2']:>10.4f} "
          f"{result['rmse_mean']:>15,.2f} {result['mae_mean']:>15,.2f}")
    print(f"{'Std Dev':<8} {result['r2_std']:>10.4f} {'—':>10} "
          f"{result['rmse_std']:>15,.2f} {result['mae_std']:>15,.2f}")

    return result


if __name__ == "__main__":
    run()
