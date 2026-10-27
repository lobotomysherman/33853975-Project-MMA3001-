"""
MMA3001 - Project 

Run with: 33853975_project.py
Dependencies: numpy, scikit-learn, pandas, pathlib and matplotlib
Outputs include regression model joblibs, results CSV files and prediction CSV files.
Docstrings use NumPy style. 
"""
import os
import csv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import time
import joblib

def load_validated_csv(csv_path):
    """
    Read a comma-separated numeric dataset and reject invalid input.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        UTF-8 CSV file with a header and at least three columns, matching
        the existing first-column exclusion and last-column target split.

    Returns
    -------
    data : pandas.DataFrame
        Finite numeric values with the original column order preserved.

    Raises
    ------
    ValueError
        If the file cannot be read, is empty, has malformed CSV structure,
        or contains missing, nonnumeric, or infinite values.

    Notes
    -----
    Validation covers every column, including the excluded first column.
    No rows are dropped and no values are imputed. Error reports identify
    one-based CSV record numbers (header is row 1), physical line spans,
    column positions, column names, and original invalid values. Blank
    records are treated as rows of missing values, not silently skipped.
    Blank cells and common NA tokens are classified as missing values.
    """
    csv_path = Path(csv_path)
    records = []
    spans = []
    reader = None
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            previous_line = 0
            for row in reader:
                records.append(row)
                spans.append((previous_line + 1, reader.line_num))
                previous_line = reader.line_num
    except FileNotFoundError as exc:
        raise ValueError(f"CSV file not found: {csv_path}") from exc
    except PermissionError as exc:
        raise ValueError(f"Permission denied reading CSV: {csv_path}") from exc
    except UnicodeError as exc:
        raise ValueError(
            f"CSV is not readable UTF-8 text: {csv_path}. "
            "Save it as CSV UTF-8 and try again."
        ) from exc
    except csv.Error as exc:
        line = reader.line_num if reader is not None else "unknown"
        raise ValueError(f"Malformed CSV near physical line {line}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Cannot read CSV {csv_path}: {exc}") from exc

    if not records or not any(cell.strip() for row in records for cell in row):
        raise ValueError(f"CSV file is empty or contains only blank values: {csv_path}")

    header = [cell.strip() for cell in records[0]]
    if len(header) < 3:
        raise ValueError(
            "CSV must have at least three comma-separated columns: the "
            "excluded first column, at least one input, and the target. "
            "Check the delimiter and header row."
        )
    if any(not name for name in header):
        columns = [i + 1 for i, name in enumerate(header) if not name]
        raise ValueError(f"Missing header names at column positions: {columns}")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate CSV column names: {duplicates}")
    if len(records) == 1:
        raise ValueError("CSV contains a header but no data rows.")

    issues = []
    numeric_rows = []
    missing_tokens = {"", "na", "n/a", "nan", "null", "none", "<na>", "#n/a"}
    for record_number, row in enumerate(records[1:], start=2):
        first_line, last_line = spans[record_number - 1]
        location = f"CSV row {record_number} (physical lines {first_line}-{last_line})"
        if not row:
            row = [""] * len(header)
        if len(row) != len(header):
            issues.append(
                f"{location}: malformed row; expected {len(header)} columns, "
                f"found {len(row)}. Check delimiters and quoting."
            )
            continue
        values = []
        for column_number, (name, raw) in enumerate(zip(header, row), start=1):
            text = raw.strip()
            problem = None
            value = np.nan
            if text.casefold() in missing_tokens:
                problem = "missing value"
            else:
                try:
                    value = float(text)
                except (ValueError, OverflowError):
                    problem = "nonnumeric or unrepresentable value"
                else:
                    if np.isnan(value):
                        problem = "missing value (NaN)"
                    elif not np.isfinite(value):
                        problem = "infinite value or numeric overflow"
            if problem:
                issues.append(
                    f"{location}, column {column_number} ({name!r}): "
                    f"{problem}; value={raw!r}"
                )
            values.append(value)
        numeric_rows.append(values)

    if issues:
        raise ValueError(
            f"CSV validation failed with {len(issues)} issue(s):\n"
            + "\n".join(issues)
            + "\nCorrect the CSV and rerun. No models were fitted."
        )
    if len(numeric_rows) < 2:
        raise ValueError("CSV needs at least two data rows for a train/test split.")
    return pd.DataFrame(numeric_rows, columns=header, dtype=float)


csv_path = Path(__file__).resolve().parent / "data" / "Concrete_Data.csv"
try:
    data = load_validated_csv(csv_path)
except ValueError as exc:
    raise SystemExit(f"Input error: {exc}") from None

data_x = data.iloc[:, 0:-1].to_numpy()
data_y = data.iloc[:, -1].to_numpy()

# quick sanity checks
print("Input Data shapes:", data_x.shape, data_y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    data_x,
    data_y,
    test_size=0.20,
    random_state=42
)

print("Training shapes:", X_train.shape, y_train.shape)
print("Testing shapes:", X_test.shape, y_test.shape)

models = []
results = []

def run_polynomial_regressions(X_train, X_test, y_train, y_test, output_dir):
    """
    Fit, time, evaluate, and export polynomial regression results.

    Parameters
    ----------
    X_train : array-like of shape (n_train_samples, n_features)
        Training input features.
    X_test : array-like of shape (n_test_samples, n_features)
        Test input features, ordered consistently with ``X_train``.
    y_train : array-like of shape (n_train_samples,)
        Training target values.
    y_test : array-like of shape (n_test_samples,)
        Test target values.
    output_dir : str or pathlib.Path
        Directory for results CSV files. Missing directories are created,
        and existing files with the same names are overwritten.

    Returns
    -------
    all_results : dict of str to pandas.DataFrame
        Results keyed by regression method. Each table contains degree,
        training MSE, test MSE, test R-squared, fitting time, and separate
        training and test prediction times in seconds.
    all_models : dict of str to list of sklearn.pipeline.Pipeline
        Fitted pipelines keyed by regression method, in ascending degree
        order. Index ``k - 1`` corresponds to polynomial degree ``k``.

    Notes
    -----
    Degree ranges are 1-4 for OLS, 1-6 for Ridge, 1-7 for Lasso, and
    1-9 for polynomial-kernel SVR. Ridge and Lasso standardise expanded
    polynomial features; SVR standardises the original input features.
    All preprocessing is fitted using training data only.

    Fitting time measures elapsed wall-clock time for ``model.fit``,
    including pipeline preprocessing. It excludes model construction,
    prediction, metric calculation, printing, and CSV export.

    Prediction times measure each complete pipeline.predict call, including
    preprocessing, for the full training or test batch. They exclude scoring
    and file export; batch sizes differ, so these are not per-sample timings.

    Results are printed to three decimal places and exported without
    rounding to one CSV file per regression method. This function does
    not perform cross-validation or hyperparameter optimisation.
    Fitted pipelines are saved as models/{method}_degree_{k}.joblib.
    Model saving is excluded from both fitting and prediction times.
    Eight input features and actual and predicted targets are exported to predictions/ under
    output_dir, with separate train and test CSVs for each method and degree.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    all_results = {}
    all_models = {}

    for name, max_degree in [
        ("OLS", 4),
        ("Ridge", 6),
        ("Lasso", 7),
        ("SVR",9)
    ]:
        results = []
        models = []

        for k in range(1, max_degree + 1):
            print(f"Fitting {name}: degree {k}...", flush=True)

            if name == "SVR":
                model = make_pipeline(
                    StandardScaler(),
                    SVR(
                        kernel="poly",
                        degree=k,
                        C=1.0,
                        epsilon=0.1,
                        gamma="scale",
                        coef0=1.0
                    )
                )

            else:
                steps = [
                    PolynomialFeatures(degree=k, include_bias=False)
                ]

                if name == "OLS":
                    steps.append(LinearRegression())

                elif name == "Ridge":
                    steps.extend([
                        StandardScaler(),
                        Ridge(alpha=0.1)
                    ])

                else:
                    steps.extend([
                        StandardScaler(),
                        Lasso(
                            alpha=0.1,
                            max_iter=100000,
                            tol=0.00001
                        )
                    ])

                model = make_pipeline(*steps)

            # Measure fitting time, including pipeline preprocessing
            start_time = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - start_time
            joblib.dump(model, models_dir / f"{name}_degree_{k}.joblib")

            print(
                f"{name}, degree {k}: fitting time = {fit_time:.3f} seconds",
                flush=True
            )

            models.append(model)

            # Predictions for training and testing data
            start_time = time.perf_counter()
            y_train_pred = model.predict(X_train)
            train_predict_time = time.perf_counter() - start_time
            start_time = time.perf_counter()
            y_test_pred = model.predict(X_test)
            test_predict_time = time.perf_counter() - start_time
            print(f"Prediction time: train = {train_predict_time:.6f} s, test = {test_predict_time:.6f} s", flush=True)

            for split, features, actual, predicted in [("train", X_train, y_train, y_train_pred), ("test", X_test, y_test, y_test_pred)]:
                pd.DataFrame(features, columns=["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age"]).assign(Actual=actual, Predicted=predicted).to_csv(
                    predictions_dir / f"{name}_degree_{k}_{split}.csv", index=False)

            results.append({
                "Degree": k,
                "Train MSE": mean_squared_error(y_train, y_train_pred),
                "Test MSE": mean_squared_error(y_test, y_test_pred),
                "Test R2": r2_score(y_test, y_test_pred),
                "Fit Time (s)": fit_time,
                "Train Predict Time (s)": train_predict_time,
                "Test Predict Time (s)": test_predict_time
            })

        results_df = pd.DataFrame(results)

        print(f"\n{name} polynomial regression")
        print(results_df.round(3).to_string(index=False))

        csv_file = output_dir / f"{name}_polynomial_results.csv"
        results_df.to_csv(csv_file, index=False)
        print(f"Saved to: {csv_file.resolve()}\n")

        all_results[name] = results_df
        all_models[name] = models

    return all_results, all_models

output_dir = Path(__file__).resolve().parent / "results"

regression_results, regression_models = run_polynomial_regressions(
    X_train, X_test, y_train, y_test, output_dir
)

