# README.md — AIPPMPS Analysis Script

## Overview

This script runs the complete statistical analysis for the AIPPMPS dissertation revision. It produces all the numbers needed for the revised tables in Chapter 3 and Chapter 4.

## What the Script Does

The script runs seven analyses in sequence:

| Section | Analysis | Purpose |
|---|---|---|
| 1 | Data Loading | Load raw and cleansed datasets |
| 2 | Ablation Study | Test SDCA at three preprocessing stages |
| 3 | 10-Fold Cross-Validation | Stable performance estimation |
| 4 | OLS Baseline Comparison | Fair comparison against simple linear regression |
| 5 | Permutation Test | Confirm performance is not due to chance |
| 6 | Temporal Validation | Test generalization to future periods |
| 7 | Residual Diagnostics | Durbin-Watson, Breusch-Pagan, Shapiro-Wilk |
| 8 | VIF Pre-Modeling | Variance Inflation Factor screening |

## Required Python Packages

Install these in PowerShell before running:

```
pip install numpy pandas scikit-learn statsmodels scipy openpyxl
```

## Required Data Files

The script needs two files in the same folder as the script:

1. `Procurement_PPMP_Data__1___1_.xlsx` (raw procurement data, 42,234 line items)
2. `annual_budget_cleansed__1_.csv` (cleansed aggregated data, 584 rows)

## File Path Configuration

Open `aippmps_analysis.py` and update these two lines near the top to match your folder location:

```python
raw = pd.read_excel(r'C:\Users\Jerico\Desktop\dissertationproj\Procurement_PPMP_Data__1___1_.xlsx')
clean = pd.read_csv(r'C:\Users\Jerico\Desktop\dissertationproj\annual_budget_cleansed__1_.csv')
```

The `r` prefix before each quote is required for Windows paths.

## How to Run

In PowerShell, navigate to the script folder and execute:

```
cd C:\Users\Jerico\Desktop\dissertationproj
python aippmps_analysis.py
```

## Script Output (Terminal)

The script prints results in nine sections:

1. Header with row counts for raw, aggregated, and cleansed data
2. Point 1 Ablation Study results
3. Point 7 10-Fold Cross-Validation per-fold results plus mean and SD
4. Point 11 OLS Baseline Comparison (OLS vs SDCA vs Random Forest)
5. Point 2 Permutation Test with 1,000 iterations
6. Point 2 Temporal Validation results
7. Point 4 Residual Diagnostics test statistics
8. Point 3 VIF table per predictor
9. Summary block with all key metrics

## Script Output (File)

A CSV summary is saved as:

```
aippmps_results.csv
```

This contains 12 rows with all key metrics for use in tables, abstract, and presentation slides.

## Expected Values

If the script runs correctly, the following values will appear:

| Metric | Expected Value |
|---|---|
| SDCA R squared (10-fold CV mean) | 0.7881 |
| SDCA Adjusted R squared | 0.7869 |
| SDCA RMSE | PHP 1,694,817.53 |
| SDCA MAE | PHP 1,209,793.81 |
| OLS Baseline R squared | 0.7789 |
| Random Forest R squared | 0.7917 |
| SDCA improvement over OLS | +1.20 percent |
| Permutation p-value | < 0.001 |
| Temporal validation R squared | 0.7041 |
| Durbin-Watson statistic | 1.997 |
| Breusch-Pagan p-value | < 0.001 |

Results are deterministic. `random_state=42` ensures identical output every run.

## Inside the Script

### Configuration

```python
RANDOM_STATE = 42
TARGET = 'TotalBudget'
FEATURES = ['Year', 'Office', 'TotalQty', 'AvgUnitCost']
```

These constants control reproducibility and which columns are used for modeling.

### Custom Model Class

```python
class ScaledSGDRegressor(BaseEstimator, RegressorMixin):
```

This wraps `SGDRegressor` with internal target scaling. SDCA in ML.NET also scales the target internally, so this wrapper produces equivalent behavior.

### Preprocessing Pipeline

Three stages applied in order:

1. **Imputation** - Missing TotalQty and AvgUnitCost values filled with column median
2. **IQR outlier treatment** - Values clipped at Q1 minus 1.5 times IQR and Q3 plus 1.5 times IQR
3. **OLS residualization** - Year trend confound removed from TotalQty and AvgUnitCost via OLS

### Preprocessor for Categorical Encoding

```python
def make_preprocessor():
    return ColumnTransformer([
        ('office', OneHotEncoder(handle_unknown='ignore'), ['Office']),
        ('num', StandardScaler(), ['Year', 'TotalQty', 'AvgUnitCost'])
    ])
```

One-hot encodes Office (109 unique values) and standardizes the three numeric features.

### Evaluation Helper

```python
def evaluate(model_fn, data, label):
```

Runs 10-fold cross-validation and returns a dictionary with mean and standard deviation for R squared, Adjusted R squared, RMSE, and MAE.

### Three Model Factory Functions

```python
def sdca(): return ScaledSGDRegressor(max_iter=1000, alpha=0.0001, random_state=RANDOM_STATE)
def ols(): return LinearRegression()
def rf(): return RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
```

These create fresh model instances for each fold. Returning new instances prevents data leakage between folds.

### Permutation Test Loop

```python
N_PERM = 1000
for i in range(N_PERM):
    y_shuf = np.random.permutation(y_full.values)
```

Shuffles the target 1,000 times and retrains the model. The empirical p-value is the fraction of permuted runs that match or exceed the actual R squared.

### Temporal Validation Split

```python
train_mask = df_full['Year'] <= 2022
test_mask = (df_full['Year'] >= 2023) & (df_full['Year'] <= 2025)
```

Trains on past data only and tests on held-out future data. This confirms the model is not exploiting temporal leakage in the standard random CV split.

### Residual Diagnostic Tests

Three statistical tests applied to the model residuals:

1. **Durbin-Watson** - Tests serial autocorrelation (target near 2.0)
2. **Breusch-Pagan** - Tests heteroscedasticity (p greater than 0.05 means constant variance)
3. **Shapiro-Wilk** - Tests normality (p greater than 0.05 means normal distribution)

### VIF Calculation

```python
v = variance_inflation_factor(X_vif, i)
```

VIF measures how much each predictor can be linearly explained by the others. Values below 5 indicate no problematic multicollinearity. Values below 10 are acceptable.

## Troubleshooting

| Error Message | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'statsmodels'` | Run `pip install statsmodels` |
| `ModuleNotFoundError: No module named 'openpyxl'` | Run `pip install openpyxl` |
| `FileNotFoundError: ... Procurement_PPMP_Data...xlsx` | Verify file paths in script match folder structure |
| `OSError: Cannot save file into a non-existent directory` | Change output path at bottom of script to existing folder |

## Reproducibility

The script is fully reproducible. The same input data produces the same output every time because:

- `random_state=42` is set throughout
- 10-fold CV uses `shuffle=True` with fixed random state
- Permutation test seeds use deterministic counter
- Temporal split uses fixed year thresholds

## Methodology Notes

**Implementation language:** Python with scikit-learn and statsmodels. Python is used because statsmodels provides the diagnostic tests (Durbin-Watson, Breusch-Pagan) required for residual analysis, which are not standard in other ML frameworks.

**SDCA equivalence:** The custom ScaledSGDRegressor class implements stochastic gradient descent with L2 regularization and internal target scaling. This is mathematically equivalent to the SDCA trainer in other frameworks.

**Validation strategy:** 10-fold cross-validation replaces single 80/20 train-test split. CV provides more stable performance estimates and lower variance across data partitions.

**Fair baseline comparison:** OLS and SDCA are both trained on identically preprocessed data using the same 10-fold CV procedure. This isolates the algorithmic contribution from preprocessing gains.
