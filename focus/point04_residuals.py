"""
point04_residuals.py

Tests whether model residuals satisfy independence, homoscedasticity,
and normality assumptions. Reports Durbin-Watson, Breusch-Pagan,
Shapiro-Wilk, and residual correlations.

Output: Table 4-RD (Residual Diagnostics).
"""
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.pipeline import Pipeline
from shared import (
    load_data, apply_full_preprocessing, make_preprocessor, sdca,
    FEATURES, TARGET
)


def run():
    print("\n" + "=" * 80)
    print("POINT 4 — RESIDUAL DIAGNOSTICS (Table 4-RD)")
    print("=" * 80)

    # Load and preprocess
    _, agg_raw, _ = load_data()
    df = apply_full_preprocessing(agg_raw)
    X = df[FEATURES]
    y = df[TARGET]

    # Fit model and compute residuals
    pipe = Pipeline([('pre', make_preprocessor()), ('model', sdca())])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    resid = y.values - preds

    # Durbin-Watson test for autocorrelation
    dw = durbin_watson(resid)
    dw_interp = 'No autocorrelation' if 1.5 < dw < 2.5 else 'Possible autocorrelation'
    print(f"\nDurbin-Watson statistic:  {dw:.3f}  ({dw_interp})")

    # Breusch-Pagan test for heteroscedasticity
    X_design = pipe.named_steps['pre'].transform(X)
    if hasattr(X_design, 'toarray'):
        X_design = X_design.toarray()
    X_design = sm.add_constant(X_design)
    bp_stat, bp_p, _, _ = het_breuschpagan(resid, X_design)
    bp_interp = 'Homoscedastic' if bp_p > 0.05 else 'Heteroscedastic'
    print(f"Breusch-Pagan chi2:       {bp_stat:.3f}  (p = {bp_p:.4f})  ({bp_interp})")

    # Residual summary
    print(f"Residual mean:            {resid.mean():,.2f}  (target ~ 0)")
    print(f"Residual std deviation:   {resid.std():,.2f}")

    # Shapiro-Wilk normality test
    sample = resid if len(resid) < 5000 else np.random.choice(resid, 5000, replace=False)
    sw_stat, sw_p = stats.shapiro(sample)
    sw_interp = 'Normal' if sw_p > 0.05 else 'Non-normal'
    print(f"Shapiro-Wilk W:           {sw_stat:.4f}  (p = {sw_p:.4f})  ({sw_interp})")

    # Residual vs Year
    r_year, p_year = stats.pearsonr(df['Year'], resid)
    print(f"Residual vs Year:         r = {r_year:.3f}  (p = {p_year:.4f})")

    # Residual vs Office mean
    office_means = df.groupby('Office')[TARGET].transform('mean')
    r_off, p_off = stats.pearsonr(office_means, resid)
    print(f"Residual vs Office mean:  r = {r_off:.3f}  (p = {p_off:.4f})")

    return {
        'durbin_watson': dw,
        'breusch_pagan_stat': bp_stat,
        'breusch_pagan_p': bp_p,
        'shapiro_w': sw_stat,
        'shapiro_p': sw_p,
        'residual_mean': resid.mean(),
        'residual_std': resid.std(),
    }


if __name__ == "__main__":
    run()
