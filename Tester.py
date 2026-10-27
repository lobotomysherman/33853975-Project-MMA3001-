import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


csv_path = Path(__file__).resolve().parent / "data" / "Concrete_Data.csv"
data = pd.read_csv(csv_path)

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

# Fit LR to model
LinReg = LinearRegression().fit(X_train, y_train)

# Target value predictions for train & test data 
LinReg_train, LinReg_test = LinReg.predict(X_train), LinReg.predict(X_test)

# Calculate model performance (MSE and R2)
print(f"Linear Regression:")
print(f"train MSE = {mean_squared_error(y_train, LinReg_train):.3f}, train R2 = {r2_score(y_train, LinReg_train):.3f}")
print(f"test MSE = {mean_squared_error(y_test, LinReg_test):.3f}, test R2 = {r2_score(y_test, LinReg_test):.3f}\n")

models = []
results = []

# Run standard LR for polynomials of degree 1-3 
for k in range(1,5):
    model = make_pipeline(PolynomialFeatures(degree=k, include_bias=False), LinearRegression())
    model.fit(X_train, y_train)
    models.append(model)

    # Target value predictions for train & test data 
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # list used to store results 
    results.append({
    "Degree": k,
    "Train MSE": mean_squared_error(y_train, y_train_pred),
    "Test MSE": mean_squared_error(y_test, y_test_pred),
    "Test R2": r2_score(y_test, y_test_pred)})

    models[k-1] = model

# DataFrame object used to present results 
PolyReg = pd.DataFrame(results)

# print results 
print('OLS polynomial regression')
print(PolyReg.round(3).to_string(index=False))

# Repeat for ridge regression 
results = []
models = []

for k in range(1,7): # Validation MSE for Ridge regressor is minimised for a higher degree vs. OLS regressor 
    model = make_pipeline(PolynomialFeatures(degree=k, include_bias=False), StandardScaler(), Ridge(alpha=0.1))
    model.fit(X_train, y_train)
    models.append(model)

    # Target value predictions for train & test data 
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # list used to store results 
    results.append({
    "Degree": k,
    "Train MSE": mean_squared_error(y_train, y_train_pred),
    "Test MSE": mean_squared_error(y_test, y_test_pred),
    "Test R2": r2_score(y_test, y_test_pred)})

    models[k-1] = model

# DataFrame object used to present results 
PolyReg = pd.DataFrame(results)

# Print results 
print('Ridge polynomial regression')
print(PolyReg.round(3).to_string(index=False))

# Repeat for Lasso regression 
results = []
models = []

for k in range(1,8): # Validation MSE for Lasso regressor is minimised for a higher degree vs. OLS regressor 
    model = make_pipeline(PolynomialFeatures(degree=k, include_bias=False), StandardScaler(), Lasso(alpha=0.1, max_iter=100000, tol=0.00001))
    model.fit(X_train, y_train)
    models.append(model)

    # Target value predictions for train & test data 
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # list used to store results 
    results.append({
    "Degree": k,
    "Train MSE": mean_squared_error(y_train, y_train_pred),
    "Test MSE": mean_squared_error(y_test, y_test_pred),
    "Test R2": r2_score(y_test, y_test_pred)})

    models[k-1] = model

# DataFrame object used to present results 
PolyReg = pd.DataFrame(results)

# Print results 
print('Lasso polynomial regression')
print(PolyReg.round(3).to_string(index=False))

models = []
results = []

# Run SVR for polynomial kernel degrees 1–9
for k in range(1, 10):
    model = make_pipeline(
        StandardScaler(),
        SVR(kernel="poly", degree=k, C=1.0, epsilon=0.1,
            gamma="scale", coef0=1.0)
    )
    model.fit(X_train, y_train)
    models.append(model)

    # Target value predictions for train & test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # List used to store results
    results.append({
        "Degree": k,
        "Train MSE": mean_squared_error(y_train, y_train_pred),
        "Test MSE": mean_squared_error(y_test, y_test_pred),
        "Test R2": r2_score(y_test, y_test_pred)
    })

    models[k-1] = model

# DataFrame object used to present results
PolyReg = pd.DataFrame(results)

# Print results
print('SVR polynomial regression')
print(PolyReg.round(3).to_string(index=False))