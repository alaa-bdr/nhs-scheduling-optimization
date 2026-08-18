"""Linear regression with all pre-operative features, one-hot encoded.

This is the proper linear baseline for the model comparison. The
baseline.py version used only 3 numeric features as a rough benchmark.
This version uses every available pre-operative feature.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA_DIR = Path("data/modelling")
TARGET = "operation_length_mins"


def build_preprocessor(x):
    numeric = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical = x.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical),
    ]), numeric, categorical


def run():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(DATA_DIR / "validation.parquet")

    y_train = train[TARGET]
    y_val = validation[TARGET]
    x_train = train.drop(columns=[TARGET])
    x_val = validation.drop(columns=[TARGET])

    for col in x_train.select_dtypes(include=[np.number]).columns:
        median = x_train[col].median()
        x_train[col] = x_train[col].fillna(median)
        x_val[col] = x_val[col].fillna(median)

    for col in x_train.select_dtypes(include=["object", "string", "category"]).columns:
        x_train[col] = x_train[col].fillna("missing").astype(str)
        x_val[col] = x_val[col].fillna("missing").astype(str)

    preprocessor, num, cat = build_preprocessor(x_train)

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", Ridge(random_state=42)),
    ])

    param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}

    print(f"Numeric features: {len(num)}, categorical: {len(cat)}")
    print(f"Running grid search with 5-fold cross-validation...")

    gs = GridSearchCV(
        pipeline, param_grid, cv=5,
        scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1,
    )
    gs.fit(x_train, y_train)

    print(f"\nBest parameters: {gs.best_params_}")
    print(f"Best CV MAE: {-gs.best_score_:.1f} mins")

    y_pred = gs.best_estimator_.predict(x_val)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    r2 = r2_score(y_val, y_pred)

    print(f"\nValidation results:")
    print(f"  MAE  = {mae:.1f} mins")
    print(f"  RMSE = {rmse:.1f} mins")
    print(f"  R2   = {r2:.3f}")

    hospital = validation.dropna(subset=["ExpectedDurationMins"])
    hospital_mae = mean_absolute_error(
        hospital[TARGET], hospital["ExpectedDurationMins"]
    )
    print(f"\nHospital planned time MAE: {hospital_mae:.1f} mins")
    print(f"Ridge regression improvement: {hospital_mae - mae:.1f} mins better")


if __name__ == "__main__":
    run()
