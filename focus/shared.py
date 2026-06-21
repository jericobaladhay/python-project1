"""
shared.py
Common code used by all reviewer-point analyses.
Contains: configuration, data loading, model wrapper, evaluation helper.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

RAW_PATH = r'C:\Users\Jerico\Desktop\dissertationproj\Procurement_PPMP_Data__1___1_.xlsx'
CLEAN_PATH = r'C:\Users\Jerico\Desktop\dissertationproj\annual_budget_cleansed__1_.csv'

TARGET = 'TotalBudget'
FEATURES = ['Year', 'Office', 'TotalQty', 'AvgUnitCost']


# ===== DATA LOADING =====
def load_data():
    """Load raw and cleansed datasets. Returns (raw, agg_raw, clean)."""
    raw = pd.read_excel(RAW_PATH)
    clean = pd.read_csv(CLEAN_PATH)

    # Aggregate raw to (Year, Office) level
    raw['TotalQty_row'] = raw[['1st', '2nd', '3rd', '4th']].sum(axis=1)
    raw['LineCost'] = raw['TotalQty_row'] * raw['UnitCost']
    agg_raw = raw.groupby(['Year', 'Office']).agg(
        TotalBudget=('LineCost', 'sum'),
        TotalQty=('TotalQty_row', 'sum'),
        AvgUnitCost=('UnitCost', 'mean')
    ).reset_index()

    return raw, agg_raw, clean


def apply_full_preprocessing(df_in):
    """Apply imputation + IQR outlier treatment + OLS residualization."""
    df = df_in.copy()
    # Imputation
    for col in ['TotalQty', 'AvgUnitCost']:
        df[col] = df[col].fillna(df[col].median())
    # IQR winsorize
    for col in ['TotalQty', 'AvgUnitCost', 'TotalBudget']:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
    # OLS residualization on Year
    for col in ['TotalQty', 'AvgUnitCost']:
        ols = LinearRegression().fit(df[['Year']], df[col])
        df[col] = df[col] - ols.predict(df[['Year']])
    return df


# ===== MODEL WRAPPER =====
class ScaledSGDRegressor(BaseEstimator, RegressorMixin):
    """SGDRegressor with internal target scaling. Mimics ML.NET SDCA behavior."""

    def __init__(self, max_iter=1000, alpha=0.0001, random_state=RANDOM_STATE):
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        self.y_scaler_ = StandardScaler()
        y_s = self.y_scaler_.fit_transform(np.asarray(y).reshape(-1, 1)).ravel()
        self.model_ = SGDRegressor(
            max_iter=self.max_iter, penalty='l2',
            alpha=self.alpha, random_state=self.random_state, tol=1e-5
        )
        self.model_.fit(X, y_s)
        return self

    def predict(self, X):
        pred_s = self.model_.predict(X)
        return self.y_scaler_.inverse_transform(pred_s.reshape(-1, 1)).ravel()


# ===== PREPROCESSOR FACTORY =====
def make_preprocessor():
    """One-hot encode Office, standardize numerics."""
    return ColumnTransformer([
        ('office', OneHotEncoder(handle_unknown='ignore'), ['Office']),
        ('num', StandardScaler(), ['Year', 'TotalQty', 'AvgUnitCost'])
    ])


# ===== MODEL FACTORIES =====
def sdca():
    return ScaledSGDRegressor()


def ols():
    return LinearRegression()


def rf():
    return RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)


# ===== EVALUATION HELPER =====
def evaluate(model_fn, data, label):
    """Run 10-fold CV. Returns dict with mean/std for R2, Adj R2, RMSE, MAE."""
    X = data[FEATURES]
    y = data[TARGET]
    pipe = Pipeline([('pre', make_preprocessor()), ('model', model_fn())])
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    r2_scores, rmse_scores, mae_scores = [], [], []
    for tr, te in kf.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        r2_scores.append(r2_score(yte, pred))
        rmse_scores.append(np.sqrt(mean_squared_error(yte, pred)))
        mae_scores.append(mean_absolute_error(yte, pred))

    n, p = len(y), len(FEATURES)
    r2_mean = np.mean(r2_scores)
    adj_r2 = 1 - (1 - r2_mean) * (n - 1) / (n - p - 1)

    return {
        'label': label,
        'r2_mean': r2_mean, 'r2_std': np.std(r2_scores),
        'adj_r2': adj_r2,
        'rmse_mean': np.mean(rmse_scores), 'rmse_std': np.std(rmse_scores),
        'mae_mean': np.mean(mae_scores), 'mae_std': np.std(mae_scores),
        'fold_r2': r2_scores, 'fold_rmse': rmse_scores, 'fold_mae': mae_scores
    }
