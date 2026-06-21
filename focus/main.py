"""
main.py
MASTER RUNNER — Executes all six reviewer-point analyses in sequence.

Run this to generate the complete revision package output.
Run individual point scripts (e.g. python point07_cv.py) to test
just one analysis at a time.
"""
import pandas as pd
import point01_ablation
import point02_permutation
import point03_vif
import point04_residuals
import point07_cv
import point11_baseline


def main():
    print("=" * 80)
    print("AIPPMPS REVISIONS — FULL ANALYSIS REPORT")
    print("=" * 80)

    # Run each reviewer-point analysis
    results = {
        'ablation': point01_ablation.run(),
        'permutation': point02_permutation.run(),
        'vif': point03_vif.run(),
        'residuals': point04_residuals.run(),
        'cv': point07_cv.run(),
        'baseline': point11_baseline.run(),
    }

    # Build summary
    print("\n" + "=" * 80)
    print("SUMMARY — VALUES FOR ABSTRACT AND TABLE 10")
    print("=" * 80)

    sdca = results['cv']
    ols = results['baseline']['ols']
    rf = results['baseline']['rf']
    perm = results['permutation']
    resid = results['residuals']

    print(f"""
SDCA primary metrics (10-fold CV):
  R2            = {sdca['r2_mean']:.4f}  (+/- {sdca['r2_std']:.4f})
  Adjusted R2   = {sdca['adj_r2']:.4f}
  RMSE          = PHP {sdca['rmse_mean']:,.2f}  (+/- {sdca['rmse_std']:,.2f})
  MAE           = PHP {sdca['mae_mean']:,.2f}

OLS baseline:
  R2            = {ols['r2_mean']:.4f}
  RMSE          = PHP {ols['rmse_mean']:,.2f}

Random Forest:
  R2            = {rf['r2_mean']:.4f}
  RMSE          = PHP {rf['rmse_mean']:,.2f}

Permutation test (N=1000):
  Null mean R2  = {perm['perm_r2_mean']:.4f}
  Actual R2     = {perm['actual_r2']:.4f}
  p-value       = {'< 0.001' if perm['p_empirical'] < 0.001 else f"{perm['p_empirical']:.4f}"}

Temporal validation:
  R2            = {perm['temp_r2']:.4f}
  RMSE          = PHP {perm['temp_rmse']:,.2f}

Residual diagnostics:
  Durbin-Watson = {resid['durbin_watson']:.3f}
  Breusch-Pagan p = {resid['breusch_pagan_p']:.4f}
""")

    # Save results CSV
    summary_rows = [
        {'metric': 'SDCA_R2', 'value': sdca['r2_mean'], 'sd': sdca['r2_std']},
        {'metric': 'SDCA_AdjR2', 'value': sdca['adj_r2'], 'sd': None},
        {'metric': 'SDCA_RMSE', 'value': sdca['rmse_mean'], 'sd': sdca['rmse_std']},
        {'metric': 'SDCA_MAE', 'value': sdca['mae_mean'], 'sd': sdca['mae_std']},
        {'metric': 'OLS_R2', 'value': ols['r2_mean'], 'sd': ols['r2_std']},
        {'metric': 'OLS_RMSE', 'value': ols['rmse_mean'], 'sd': ols['rmse_std']},
        {'metric': 'RF_R2', 'value': rf['r2_mean'], 'sd': rf['r2_std']},
        {'metric': 'RF_RMSE', 'value': rf['rmse_mean'], 'sd': rf['rmse_std']},
        {'metric': 'Permutation_p', 'value': perm['p_empirical'], 'sd': None},
        {'metric': 'Temporal_R2', 'value': perm['temp_r2'], 'sd': None},
        {'metric': 'DurbinWatson', 'value': resid['durbin_watson'], 'sd': None},
        {'metric': 'BreuschPagan_p', 'value': resid['breusch_pagan_p'], 'sd': None},
    ]
    pd.DataFrame(summary_rows).to_csv('aippmps_results.csv', index=False)
    print("Results saved to: aippmps_results.csv")


if __name__ == "__main__":
    main()
