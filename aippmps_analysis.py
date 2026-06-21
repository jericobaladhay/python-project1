"""
AIPPMPS Dissertation Revisions — Master Analysis Script

"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("="*80)
print("AIPPMPS REVISIONS — ANALYSIS REPORT")
print("="*80)

# raw = pd.read_excel('/mnt/user-data/uploads/Procurement_PPMP_Data__1___1_.xlsx')
# clean = pd.read_csv('/mnt/user-data/uploads/annual_budget_cleansed__1_.csv')
raw = pd.read_excel(r'C:\Users\Jerico\Desktop\dissertationproj\Procurement_PPMP_Data__1___1_.xlsx')
clean = pd.read_csv(r'C:\Users\Jerico\Desktop\dissertationproj\annual_budget_cleansed__1_.csv')

# Aggregate raw to (Year, Office) level — matches dissertation methodology
raw['TotalQty_row'] = raw[['1st','2nd','3rd','4th']].sum(axis=1)
raw['LineCost'] = raw['TotalQty_row'] * raw['UnitCost']
agg_raw = raw.groupby(['Year','Office']).agg(
    TotalBudget=('LineCost','sum'),
    TotalQty=('TotalQty_row','sum'),
    AvgUnitCost=('UnitCost','mean')
).reset_index()

print(f"\nRaw line items:       {len(raw):,}")
print(f"Aggregated raw rows:  {len(agg_raw):,}")
print(f"Cleansed rows:        {len(clean):,}")

# Use the cleansed dataset as the modeling base (matches your dissertation)
df = clean.copy()
TARGET = 'TotalBudget'
FEATURES = ['Year', 'Office', 'TotalQty', 'AvgUnitCost']

# ============================================================================
# 2. PREPROCESSING PIPELINES (for Point 1: Ablation Study)
# ============================================================================
def make_preprocessor():
    """One-hot encode Office; standardize numerics. SGD requires scaling."""
    return ColumnTransformer([
        ('office', OneHotEncoder(handle_unknown='ignore'), ['Office']),
        ('num', StandardScaler(), ['Year', 'TotalQty', 'AvgUnitCost'])
    ])

# Helper: scaled SDCA wrapper that also scales the target (critical for SGD)
from sklearn.base import BaseEstimator, RegressorMixin
class ScaledSGDRegressor(BaseEstimator, RegressorMixin):
    """SGDRegressor with internal target scaling — mimics ML.NET SDCA behavior."""
    def __init__(self, max_iter=1000, alpha=0.0001, random_state=42):
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state
    def fit(self, X, y):
        self.y_scaler_ = StandardScaler()
        y_s = self.y_scaler_.fit_transform(np.asarray(y).reshape(-1,1)).ravel()
        self.model_ = SGDRegressor(max_iter=self.max_iter, penalty='l2',
                                    alpha=self.alpha, random_state=self.random_state,
                                    tol=1e-5)
        self.model_.fit(X, y_s)
        return self
    def predict(self, X):
        pred_s = self.model_.predict(X)
        return self.y_scaler_.inverse_transform(pred_s.reshape(-1,1)).ravel()

# Stage (a): Raw — start from aggregated raw, no treatment
df_raw_stage = agg_raw.copy()

# Stage (b): Imputation only (raw with median imputation; here data is complete
#            but we simulate imputation step for the pipeline)
df_imputed = agg_raw.copy()
for col in ['TotalQty', 'AvgUnitCost']:
    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

# Stage (c): Full preprocessing — imputation + IQR outlier treatment + OLS residualization
df_full = df_imputed.copy()

# IQR outlier treatment (winsorize at IQR fences)
for col in ['TotalQty', 'AvgUnitCost', 'TotalBudget']:
    Q1, Q3 = df_full[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    df_full[col] = df_full[col].clip(lower=lo, upper=hi)

# OLS residualization on Year (Stage 3 in the solutions doc)
for col in ['TotalQty', 'AvgUnitCost']:
    ols_year = LinearRegression().fit(df_full[['Year']], df_full[col])
    df_full[col] = df_full[col] - ols_year.predict(df_full[['Year']])

# ============================================================================
# Helper: fit & score a model, return all metrics
# ============================================================================
def evaluate(model_fn, data, label):
    """Run 10-fold CV, return mean/std for R2, Adj R2, RMSE, MAE."""
    X = data[FEATURES]
    y = data[TARGET]
    pre = make_preprocessor()
    pipe = Pipeline([('pre', pre), ('model', model_fn())])
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    r2_scores, rmse_scores, mae_scores = [], [], []
    for tr_idx, te_idx in kf.split(X):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        r2_scores.append(r2_score(yte, pred))
        rmse_scores.append(np.sqrt(mean_squared_error(yte, pred)))
        mae_scores.append(mean_absolute_error(yte, pred))

    n, p = len(y), len(FEATURES)
    r2_mean = np.mean(r2_scores)
    adj_r2 = 1 - (1-r2_mean)*(n-1)/(n-p-1)
    return {
        'label': label,
        'r2_mean': r2_mean, 'r2_std': np.std(r2_scores),
        'adj_r2': adj_r2,
        'rmse_mean': np.mean(rmse_scores), 'rmse_std': np.std(rmse_scores),
        'mae_mean': np.mean(mae_scores),  'mae_std': np.std(mae_scores),
        'fold_r2': r2_scores, 'fold_rmse': rmse_scores, 'fold_mae': mae_scores
    }

def sdca(): return ScaledSGDRegressor(max_iter=1000, alpha=0.0001,
                                       random_state=RANDOM_STATE)
def ols():  return LinearRegression()
def rf():   return RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)

# ============================================================================
# 3. POINT 1 — ABLATION STUDY (Table 3-B)
# ============================================================================
print("\n" + "="*80)
print("POINT 1 — DATA PREPROCESSING ABLATION STUDY (Table 3-B)")
print("="*80)
abl_raw   = evaluate(sdca, df_raw_stage, '(a) Raw, unprocessed')
abl_imp   = evaluate(sdca, df_imputed,   '(b) Imputation only')
abl_full  = evaluate(sdca, df_full,      '(c) Full preprocessing')

print(f"\n{'Condition':<30} {'R²':>10} {'Adj R²':>10} {'RMSE (PHP)':>15}")
print("-"*70)
for r in [abl_raw, abl_imp, abl_full]:
    print(f"{r['label']:<30} {r['r2_mean']:>10.4f} {r['adj_r2']:>10.4f} {r['rmse_mean']:>15,.2f}")

# Data quality metrics (Table 3-A)
print("\nTable 3-A — Data Quality Metrics:")
print(f"  Pre-processing  AvgUnitCost CV: {agg_raw['AvgUnitCost'].std()/agg_raw['AvgUnitCost'].mean():.2f}")
print(f"  Post-processing AvgUnitCost CV: {df_full['AvgUnitCost'].std()/(df_imputed['AvgUnitCost'].mean()):.2f}")
print(f"  Pre-processing  TotalQty CV:    {agg_raw['TotalQty'].std()/agg_raw['TotalQty'].mean():.2f}")
print(f"  Post-processing TotalQty CV:    {df_full['TotalQty'].std()/(df_imputed['TotalQty'].mean()+1e-9):.2f}")
r_year_qty_pre  = stats.pearsonr(agg_raw['Year'], agg_raw['TotalQty'])[0]
r_year_qty_post = stats.pearsonr(df_full['Year'], df_full['TotalQty'])[0]
print(f"  Year–TotalQty R² pre:  {r_year_qty_pre**2:.3f}")
print(f"  Year–TotalQty R² post: {r_year_qty_post**2:.3f}  (residualization check)")

# ============================================================================
# 4. POINT 7 — 10-FOLD CV ON SDCA (Table 4-CV)
# ============================================================================
print("\n" + "="*80)
print("POINT 7 — 10-FOLD CROSS-VALIDATION FOR SDCA (Table 4-CV)")
print("="*80)
sdca_cv = abl_full  # already computed on full preprocessing
print(f"\n{'Fold':<8} {'R²':>10} {'Adj R²':>10} {'RMSE':>15} {'MAE':>15}")
print("-"*65)
for i, (r2, rmse, mae) in enumerate(zip(sdca_cv['fold_r2'],
                                         sdca_cv['fold_rmse'],
                                         sdca_cv['fold_mae']), 1):
    n, p = len(df_full), len(FEATURES)
    adj = 1 - (1-r2)*(n-1)/(n-p-1)
    print(f"Fold {i:<3} {r2:>10.4f} {adj:>10.4f} {rmse:>15,.2f} {mae:>15,.2f}")
print("-"*65)
print(f"{'Mean':<8} {sdca_cv['r2_mean']:>10.4f} {sdca_cv['adj_r2']:>10.4f} "
      f"{sdca_cv['rmse_mean']:>15,.2f} {sdca_cv['mae_mean']:>15,.2f}")
print(f"{'Std Dev':<8} {sdca_cv['r2_std']:>10.4f} {'—':>10} "
      f"{sdca_cv['rmse_std']:>15,.2f} {sdca_cv['mae_std']:>15,.2f}")

# ============================================================================
# 5. POINT 11 — OLS BASELINE COMPARISON (Table 4-D)
# ============================================================================
print("\n" + "="*80)
print("POINT 11 — OLS BASELINE vs SDCA (Table 4-D)")
print("="*80)
ols_cv = evaluate(ols, df_full, 'OLS Baseline')
rf_cv  = evaluate(rf,  df_full, 'Random Forest')
print(f"\n{'Model':<25} {'R²':>10} {'Adj R²':>10} {'RMSE':>15} {'MAE':>15}")
print("-"*80)
for r in [ols_cv, sdca_cv, rf_cv]:
    print(f"{r['label']:<25} {r['r2_mean']:>10.4f} {r['adj_r2']:>10.4f} "
          f"{r['rmse_mean']:>15,.2f} {r['mae_mean']:>15,.2f}")
improv_r2  = (sdca_cv['r2_mean'] - ols_cv['r2_mean']) / abs(ols_cv['r2_mean']) * 100 if ols_cv['r2_mean']!=0 else float('nan')
improv_abs = sdca_cv['r2_mean'] - ols_cv['r2_mean']
print(f"\nSDCA vs OLS:  ΔR² = {improv_abs:+.4f}  ({improv_r2:+.1f}%)")

# ============================================================================
# 6. POINT 2 — PERMUTATION TEST + TEMPORAL VALIDATION
# ============================================================================
print("\n" + "="*80)
print("POINT 2 — PERMUTATION TEST (N=1000) & TEMPORAL VALIDATION")
print("="*80)
X_full = df_full[FEATURES]
y_full = df_full[TARGET]
pre = make_preprocessor()
pipe = Pipeline([('pre', pre), ('model', sdca())])

# Actual score (single 80/20 to match dissertation framing)
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X_full, y_full, test_size=0.2,
                                      random_state=RANDOM_STATE)
pipe.fit(Xtr, ytr)
actual_r2 = r2_score(yte, pipe.predict(Xte))
print(f"\nActual SDCA R² (80/20): {actual_r2:.4f}")

# Permutation test
N_PERM = 1000
perm_r2 = []
print(f"Running {N_PERM} permutations...")
for i in range(N_PERM):
    y_shuf = np.random.permutation(y_full.values)
    Xtr_p, Xte_p, ytr_p, yte_p = train_test_split(X_full, y_shuf, test_size=0.2,
                                                   random_state=i)
    p = Pipeline([('pre', make_preprocessor()), ('model', sdca())])
    p.fit(Xtr_p, ytr_p)
    perm_r2.append(r2_score(yte_p, p.predict(Xte_p)))
    if (i+1) % 200 == 0:
        print(f"  ...{i+1}/{N_PERM}")

perm_r2 = np.array(perm_r2)
p_empirical = (perm_r2 >= actual_r2).sum() / N_PERM
print(f"\nPermuted null R²: mean={perm_r2.mean():.4f}, SD={perm_r2.std():.4f}")
print(f"Empirical p-value: {p_empirical:.4f}  ({'p < 0.001' if p_empirical < 0.001 else f'p = {p_empirical:.4f}'})")

# Temporal validation: train on 2017-2022, test on 2023-2025
train_mask = df_full['Year'] <= 2022
test_mask  = (df_full['Year'] >= 2023) & (df_full['Year'] <= 2025)
Xtr_t, Xte_t = X_full[train_mask], X_full[test_mask]
ytr_t, yte_t = y_full[train_mask], y_full[test_mask]
pipe_t = Pipeline([('pre', make_preprocessor()), ('model', sdca())])
pipe_t.fit(Xtr_t, ytr_t)
pred_t = pipe_t.predict(Xte_t)
temp_r2   = r2_score(yte_t, pred_t)
temp_rmse = np.sqrt(mean_squared_error(yte_t, pred_t))
n_t = len(yte_t)
temp_adj  = 1 - (1-temp_r2)*(n_t-1)/(n_t-len(FEATURES)-1)
print(f"\nTemporal validation (train 2017-22, test 2023-25):")
print(f"  R² = {temp_r2:.4f}, Adj R² = {temp_adj:.4f}, RMSE = {temp_rmse:,.2f}")
print(f"  n_train = {train_mask.sum()}, n_test = {test_mask.sum()}")

# ============================================================================
# 7. POINT 4 — RESIDUAL DIAGNOSTICS
# ============================================================================
print("\n" + "="*80)
print("POINT 4 — RESIDUAL DIAGNOSTICS (Table 4-RD)")
print("="*80)
# Refit on full data and compute residuals
pipe_full = Pipeline([('pre', make_preprocessor()), ('model', sdca())])
pipe_full.fit(X_full, y_full)
preds = pipe_full.predict(X_full)
resid = y_full.values - preds

# Durbin-Watson
dw = durbin_watson(resid)
print(f"Durbin-Watson statistic:  {dw:.3f}  "
      f"({'No autocorrelation' if 1.5 < dw < 2.5 else 'Possible autocorrelation'})")

# Breusch-Pagan (needs design matrix)
X_design = pipe_full.named_steps['pre'].transform(X_full)
if hasattr(X_design, 'toarray'):
    X_design = X_design.toarray()
X_design = sm.add_constant(X_design)
try:
    bp_stat, bp_p, _, _ = het_breuschpagan(resid, X_design)
    print(f"Breusch-Pagan χ²:         {bp_stat:.3f}  (p = {bp_p:.4f})  "
          f"({'Homoscedastic' if bp_p > 0.05 else 'Heteroscedastic'})")
except Exception as e:
    print(f"Breusch-Pagan: could not compute ({e})")

# Residual summaries
print(f"Residual mean:            {resid.mean():,.2f}  (target ≈ 0)")
print(f"Residual std dev:         {resid.std():,.2f}")

# Shapiro-Wilk (on a sample if n is large)
sample = resid if len(resid) < 5000 else np.random.choice(resid, 5000, replace=False)
sw_stat, sw_p = stats.shapiro(sample)
print(f"Shapiro-Wilk W:           {sw_stat:.4f}  (p = {sw_p:.4f})  "
      f"({'Normal' if sw_p > 0.05 else 'Non-normal (note in writeup)'})")

# Residual vs Year correlation
r_year, p_year = stats.pearsonr(df_full['Year'], resid)
print(f"Residual vs Year:         r = {r_year:.3f}  (p = {p_year:.4f})")

# Residual vs Office (one-way ANOVA approximation via correlation with encoded mean)
office_means = df_full.groupby('Office')[TARGET].transform('mean')
r_off, p_off = stats.pearsonr(office_means, resid)
print(f"Residual vs Office mean:  r = {r_off:.3f}  (p = {p_off:.4f})")

# ============================================================================
# 8. POINT 3 — VIF (pre-modeling diagnostic on numeric features)
# ============================================================================
print("\n" + "="*80)
print("POINT 3 — VIF PRE-MODELING DIAGNOSTIC (Table 3-VIF)")
print("="*80)
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Encode Office as integer for VIF (not OHE — VIF on dummy explosion is uninformative)
df_vif = df_full.copy()
df_vif['Office_enc'] = pd.Categorical(df_vif['Office']).codes
vif_features = ['Year', 'Office_enc', 'TotalQty', 'AvgUnitCost']
X_vif = df_vif[vif_features].values
print(f"\n{'Predictor':<20} {'VIF':>8} {'Decision':<15}")
print("-"*45)
for i, feat in enumerate(vif_features):
    v = variance_inflation_factor(X_vif, i)
    print(f"{feat:<20} {v:>8.2f} {'Retained' if v < 10 else 'Drop':<15}")

# ============================================================================
# 9. FINAL SUMMARY TABLE FOR ABSTRACT
# ============================================================================
print("\n" + "="*80)
print("SUMMARY — VALUES FOR ABSTRACT AND TABLE 10")
print("="*80)
print(f"""
SDCA primary metrics (10-fold CV):
  R²            = {sdca_cv['r2_mean']:.4f}  (± {sdca_cv['r2_std']:.4f})
  Adjusted R²   = {sdca_cv['adj_r2']:.4f}
  RMSE          = PHP {sdca_cv['rmse_mean']:,.2f}  (± {sdca_cv['rmse_std']:,.2f})
  MAE           = PHP {sdca_cv['mae_mean']:,.2f}

OLS baseline (10-fold CV):
  R²            = {ols_cv['r2_mean']:.4f}
  RMSE          = PHP {ols_cv['rmse_mean']:,.2f}

Random Forest (10-fold CV):
  R²            = {rf_cv['r2_mean']:.4f}
  RMSE          = PHP {rf_cv['rmse_mean']:,.2f}

Permutation test (N=1000):
  Null mean R²  = {perm_r2.mean():.4f}
  Actual R²     = {actual_r2:.4f}
  p-value       = {'< 0.001' if p_empirical < 0.001 else f'{p_empirical:.4f}'}

Temporal validation:
  R²            = {temp_r2:.4f}
  RMSE          = PHP {temp_rmse:,.2f}

Residual diagnostics:
  Durbin-Watson = {dw:.3f}
  Breusch-Pagan p = {bp_p:.4f}
""")

# Save a tidy CSV of results
out = pd.DataFrame([
    {'metric':'SDCA_R2', 'value':sdca_cv['r2_mean'], 'sd':sdca_cv['r2_std']},
    {'metric':'SDCA_AdjR2', 'value':sdca_cv['adj_r2'], 'sd':None},
    {'metric':'SDCA_RMSE', 'value':sdca_cv['rmse_mean'], 'sd':sdca_cv['rmse_std']},
    {'metric':'SDCA_MAE', 'value':sdca_cv['mae_mean'], 'sd':sdca_cv['mae_std']},
    {'metric':'OLS_R2', 'value':ols_cv['r2_mean'], 'sd':ols_cv['r2_std']},
    {'metric':'OLS_RMSE', 'value':ols_cv['rmse_mean'], 'sd':ols_cv['rmse_std']},
    {'metric':'RF_R2', 'value':rf_cv['r2_mean'], 'sd':rf_cv['r2_std']},
    {'metric':'RF_RMSE', 'value':rf_cv['rmse_mean'], 'sd':rf_cv['rmse_std']},
    {'metric':'Permutation_p', 'value':p_empirical, 'sd':None},
    {'metric':'Temporal_R2', 'value':temp_r2, 'sd':None},
    {'metric':'DurbinWatson', 'value':dw, 'sd':None},
    {'metric':'BreuschPagan_p', 'value':bp_p, 'sd':None},
])
# out.to_csv('/mnt/user-data/outputs/aippmps_results.csv', index=False)
out.to_csv('aippmps_results.csv', index=False)
print("Results saved to: /mnt/user-data/outputs/aippmps_results.csv")
