"""
point11_baseline.py


Tests whether SDCA provides meaningful improvement over a simpler
OLS baseline trained on the same preprocessed data. Also includes
Random Forest as a non-linear baseline.

Output: Table 4-D (OLS baseline comparison).
This is the test that proves the original 51 percent claim was unfair.
"""
from shared import load_data, apply_full_preprocessing, evaluate, sdca, ols, rf


def run():
    print("\n" + "=" * 80)
    print("POINT 11 — OLS BASELINE vs SDCA (Table 4-D)")
    print("=" * 80)

    # Load and preprocess
    _, agg_raw, _ = load_data()
    df = apply_full_preprocessing(agg_raw)

    # Evaluate all three models on the SAME preprocessed data
    ols_cv = evaluate(ols, df, 'OLS Baseline')
    sdca_cv = evaluate(sdca, df, 'SDCA')
    rf_cv = evaluate(rf, df, 'Random Forest')

    # Print comparison
    print(f"\n{'Model':<25} {'R2':>10} {'Adj R2':>10} {'RMSE':>15} {'MAE':>15}")
    print("-" * 80)
    for r in [ols_cv, sdca_cv, rf_cv]:
        print(f"{r['label']:<25} {r['r2_mean']:>10.4f} {r['adj_r2']:>10.4f} "
              f"{r['rmse_mean']:>15,.2f} {r['mae_mean']:>15,.2f}")

    # Show improvement
    improv_abs = sdca_cv['r2_mean'] - ols_cv['r2_mean']
    improv_pct = (improv_abs / abs(ols_cv['r2_mean']) * 100) if ols_cv['r2_mean'] != 0 else float('nan')
    print(f"\nSDCA vs OLS:  Delta R2 = {improv_abs:+.4f}  ({improv_pct:+.1f}%)")

    return {'ols': ols_cv, 'sdca': sdca_cv, 'rf': rf_cv}


if __name__ == "__main__":
    run()
