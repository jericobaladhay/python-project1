# AIPPMPS Analysis Scripts

This folder contains the analysis split into focused, easy-to-read files.
Each file addresses one of Dr. Raga's review points.

## File Structure

| File | Purpose | Dr. Raga Point |
|------|---------|----------------|
| `shared.py` | Common code (config, data loading, model wrapper, evaluator) | Used by all files |
| `point01_ablation.py` | Ablation study (raw vs imputation vs full preprocessing) | Point 1 |
| `point02_permutation.py` | Permutation test + temporal validation | Point 2 |
| `point03_vif.py` | VIF as pre-modeling diagnostic | Point 3 |
| `point04_residuals.py` | Durbin-Watson, Breusch-Pagan, Shapiro-Wilk | Point 4 |
| `point07_cv.py` | 10-fold cross-validation | Point 7 (CRITICAL) |
| `point11_baseline.py` | OLS baseline comparison | Point 11 (CRITICAL) |
| `main.py` | Runs all six analyses and saves summary CSV | All points |

## How to Run

### Run everything
```
python main.py
```

### Run just one analysis (for testing or debugging)
```
python point07_cv.py
python point11_baseline.py
python point04_residuals.py
```

## File Paths

Before running, make sure the paths in `shared.py` point to your data files:

```python
RAW_PATH = r'C:\Users\Jerico\Desktop\dissertationproj\Procurement_PPMP_Data__1___1_.xlsx'
CLEAN_PATH = r'C:\Users\Jerico\Desktop\dissertationproj\annual_budget_cleansed__1_.csv'
```

Update these to match where your Excel and CSV files are saved.

## Required Python Packages

```
pip install numpy pandas scikit-learn statsmodels scipy openpyxl
```

## What Each File Does in Plain Words

**shared.py** - Like a toolbox. Other files borrow tools from here.

**point01_ablation.py** - Tests model at three preprocessing levels to prove the preprocessing pipeline is legitimate.

**point02_permutation.py** - Shuffles the data 1000 times to prove the model's score is not luck.

**point03_vif.py** - Checks that predictors don't overlap too much (no multicollinearity).

**point04_residuals.py** - Checks that the model's errors are random and well-behaved.

**point07_cv.py** - Trains and tests the model 10 times to confirm stable performance.

**point11_baseline.py** - Compares SDCA against simple OLS regression to see if ML actually helps.

**main.py** - The conductor. Runs all six tests and gives you the final summary.
