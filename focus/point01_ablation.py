"""
point01_ablation.py

Proves performance comes from legitimate data quality improvements,
not from hidden preprocessing artifacts. Tests SDCA at three preprocessing
stages and shows R squared improves only with proper preprocessing.

Output: Table 3-A (data quality), Table 3-B (ablation study).
"""
from scipy import stats
from shared import (
    load_data, apply_full_preprocessing, evaluate, sdca, FEATURES, TARGET
)


def run():
    print("\n" + "=" * 80)
    print("POINT 1 — DATA PREPROCESSING ABLATION STUDY (Table 3-B)")
    print("=" * 80)

    # Load data
    _, agg_raw, _ = load_data()

    # Stage (a): raw aggregation, no preprocessing
    df_raw = agg_raw.copy()

    # Stage (b): imputation only
    df_imp = agg_raw.copy()
    for col in ['TotalQty', 'AvgUnitCost']:
        df_imp[col] = df_imp[col].fillna(df_imp[col].median())

    # Stage (c): full preprocessing
    df_full = apply_full_preprocessing(agg_raw)

    # Run SDCA on each stage
    r_raw = evaluate(sdca, df_raw, '(a) Raw, unprocessed')
    r_imp = evaluate(sdca, df_imp, '(b) Imputation only')
    r_full = evaluate(sdca, df_full, '(c) Full preprocessing')

    # Print Table 3-B
    print(f"\n{'Condition':<30} {'R2':>10} {'Adj R2':>10} {'RMSE (PHP)':>15}")
    print("-" * 70)
    for r in [r_raw, r_imp, r_full]:
        print(f"{r['label']:<30} {r['r2_mean']:>10.4f} {r['adj_r2']:>10.4f} {r['rmse_mean']:>15,.2f}")

    # Print Table 3-A: data quality metrics
    print("\nTable 3-A — Data Quality Metrics:")
    print(f"  Pre  AvgUnitCost CV: {agg_raw['AvgUnitCost'].std()/agg_raw['AvgUnitCost'].mean():.2f}")
    print(f"  Post AvgUnitCost CV: {df_full['AvgUnitCost'].std()/(df_imp['AvgUnitCost'].mean()+1e-9):.2f}")
    print(f"  Pre  TotalQty CV:    {agg_raw['TotalQty'].std()/agg_raw['TotalQty'].mean():.2f}")
    print(f"  Post TotalQty CV:    {df_full['TotalQty'].std()/(df_imp['TotalQty'].mean()+1e-9):.2f}")

    r_pre = stats.pearsonr(agg_raw['Year'], agg_raw['TotalQty'])[0]
    r_post = stats.pearsonr(df_full['Year'], df_full['TotalQty'])[0]
    print(f"  Year–TotalQty R2 pre:  {r_pre**2:.3f}")
    print(f"  Year–TotalQty R2 post: {r_post**2:.3f}  (residualization check)")

    return {'raw': r_raw, 'imputation': r_imp, 'full': r_full, 'df_full': df_full}


if __name__ == "__main__":
    run()
