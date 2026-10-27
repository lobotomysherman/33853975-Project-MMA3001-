# MMA3001 Project: Polynomial Regression for Concrete Compressive Strength

This project compares four regression methods, each fitted at several polynomial degrees, for predicting concrete compressive strength from its mix composition and age. For every method and degree, the script records training and test error, test R², and timing, and it saves the fitted model and its predictions.

## Methods

All models use an 80/20 train/test split (`random_state=42`). Every preprocessing step is fitted on the training data only.

| Method | Degrees | Pipeline | Settings |
|---|---|---|---|
| OLS | 1-4 | `PolynomialFeatures` → `LinearRegression` | defaults |
| Ridge | 1-6 | `PolynomialFeatures` → `StandardScaler` → `Ridge` | `alpha=0.1` |
| Lasso | 1-7 | `PolynomialFeatures` → `StandardScaler` → `Lasso` | `alpha=0.1`, `max_iter=100000`, `tol=1e-5` |
| SVR | 1-9 | `StandardScaler` → `SVR` (polynomial kernel) | `C=1.0`, `epsilon=0.1`, `gamma="scale"`, `coef0=1.0` |

In total, 26 models are fitted (4 + 6 + 7 + 9). The script does not perform cross-validation or hyperparameter tuning.

## Repository structure

```
.
├── MMA3001_33853975_Project.py        # main script
├── Tester.py                  # independent cross-check of the main script's results
├── data/
│   └── Concrete_Data.csv      # input dataset (you must provide this)
└── results/                   # created on first run
    ├── OLS_polynomial_results.csv
    ├── Ridge_polynomial_results.csv
    ├── Lasso_polynomial_results.csv
    ├── SVR_polynomial_results.csv
    ├── models/
    │   └── {method}_degree_{k}.joblib
    └── predictions/
        └── {method}_degree_{k}_{train|test}.csv
```

## Requirements

- Python 3.9 or newer
- numpy
- pandas
- scikit-learn
- matplotlib (imported by the script, not currently used for any output)
- joblib (installed alongside scikit-learn)

Install with:

```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

## Input data

The script reads `data/Concrete_Data.csv`, located relative to the script file, so it can be run from any working directory.

The CSV must be UTF-8 with a header row. The **last column is the target** and **all preceding columns are input features**. The prediction exports label the features with these eight names, in this order, so the CSV should have exactly eight feature columns followed by the target:

1. Cement
2. Blast Furnace Slag
3. Fly Ash
4. Water
5. Superplasticizer
6. Coarse Aggregate
7. Fine Aggregate
8. Age

These match the UCI Concrete Compressive Strength dataset (https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), whose target is compressive strength.

### Input validation

Before any model is fitted, the CSV is checked and rejected if it has any of the following:

- unreadable, non-UTF-8 or malformed CSV structure
- fewer than three columns, blank or duplicate header names
- no data rows, or fewer than two
- rows with the wrong number of columns
- blank cells or common missing-value tokens (`NA`, `N/A`, `NaN`, `null`, `none`, `<NA>`, `#N/A`)
- non-numeric, infinite or overflowing values

No rows are dropped and no values are imputed. If validation fails, the script exits with an `Input error:` message listing each problem with its CSV row number, physical line span, column position, column name and original value. No models are fitted in that case.

## Usage

From the project root:

```bash
python MMA3001_33853975_Project.py
```

The script has no command-line arguments, and the paths and hyperparameters are set in the code. It runs top to bottom on execution, so importing it as a module will also trigger a full run.

Progress is printed as each model is fitted, followed by a results table for each method.

## Outputs

Everything is written under `results/`. Existing files with the same names are overwritten.

### Results tables

`results/{method}_polynomial_results.csv` contains one row per degree with these columns:

| Column | Description |
|---|---|
| `Degree` | Polynomial degree |
| `Train MSE` | Mean squared error on the training set |
| `Test MSE` | Mean squared error on the test set |
| `Test R2` | R² on the test set |
| `Fit Time (s)` | Wall-clock time of `model.fit`, including preprocessing |
| `Train Predict Time (s)` | Time for one `predict` call over the full training set |
| `Test Predict Time (s)` | Time for one `predict` call over the full test set |

Values are saved unrounded (the console output is rounded to three decimal places). Prediction times cover whole batches, so they are not per-sample timings. Model saving, scoring and file export are excluded from all timings.

### Saved models

`results/models/{method}_degree_{k}.joblib` holds each fitted pipeline. Load one and predict with:

```python
import joblib

model = joblib.load("results/models/Ridge_degree_3.joblib")
y_pred = model.predict(X_new)  # X_new: array of shape (n_samples, 8)
```

Because each file is a complete pipeline, any scaling and polynomial expansion is applied automatically. Load models with the same scikit-learn version used to create them.

### Predictions

`results/predictions/{method}_degree_{k}_train.csv` and `..._test.csv` each contain the eight input features plus `Actual` and `Predicted` columns.

## Verification with `Tester.py`

`Tester.py` is a standalone, simplified script used to cross-check the numbers produced by the main script. It is not needed to produce any of the project outputs.

It reads the same `data/Concrete_Data.csv`, uses the same 80/20 split (`random_state=42`), and fits the same models with the same hyperparameters (OLS degrees 1-4, Ridge 1-6, Lasso 1-7, SVR 1-9). It is written as a straight sequence of per-method loops rather than the main script's single function.

Run it from the project root:

```bash
python Tester.py
```

It prints to the console only and writes no files. For each method it prints a table of `Degree`, `Train MSE`, `Test MSE` and `Test R2`, rounded to three decimal places. It also prints train and test MSE and R² for a plain (non-polynomial) `LinearRegression` as a baseline, along with the data shapes.

Compared with the main script, it:

- reads the CSV with `pandas.read_csv`, without the input validation
- records no timings
- saves no models or predictions

To check the main script's results, run both scripts and compare the printed tables against the matching columns in `results/{method}_polynomial_results.csv`. The error and R² values should agree to three decimal places. The timing columns exist only in the main script's output.

## Notes

- `{method}` is one of `OLS`, `Ridge`, `Lasso` or `SVR`, and `{k}` is the polynomial degree.
- For SVR, the standardisation is applied to the original input features, since the polynomial expansion is handled by the kernel.
- Functions use NumPy-style docstrings.
