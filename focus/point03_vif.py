"""
point03_vif.py
DR. RAGA POINT 3: VIF as Pre-Modeling Diagnostic

Reports Variance Inflation Factor BEFORE modeling, not as post-hoc
justification. VIF screens for multicollinearity in predictors.

Output: Table 3-VIF (Variance Inflation Factor).
"""
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from shared import load_data, apply_full_preprocessing


def run():
    print("\n" + "=" * 80)
    print("POINT 3 — VIF PRE-MODELING DIAGNOSTIC (Table 3-VIF)")
    print("=" * 80)

    # Load and preprocess
    _, agg_raw, _ = load_data()
    df = apply_full_preprocessing(agg_raw)

    # Encode Office as integer codes (OHE explosion uninformative for VIF)
    df['Office_enc'] = pd.Categorical(df['Office']).codes
    vif_features = ['Year', 'Office_enc', 'TotalQty', 'AvgUnitCost']
    X = df[vif_features].values

    # Compute VIF for each feature
    print(f"\n{'Predictor':<20} {'VIF':>8} {'Decision':<15}")
    print("-" * 45)
    results = {}
    for i, feat in enumerate(vif_features):
        v = variance_inflation_factor(X, i)
        decision = 'Retained' if v < 10 else 'Drop'
        print(f"{feat:<20} {v:>8.2f} {decision:<15}")
        results[feat] = v

    return results


if __name__ == "__main__":
    run()
