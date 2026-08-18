"""XGBoost regression for operation duration prediction.

Uses cross-validation and grid search to find the best hyperparameters,
then evaluates on the validation set against the hospital baseline.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path("data/modelling")
TARGET = "operation_length_mins"


def encode_categoricals(train, validation):
    """Label-encode string columns, handling unseen values in validation."""
    encoders = {}
    train_enc = train.copy()
    val_enc = validation.copy()

    for col in train.select_dtypes(include=["object", "string", "category"]).columns:
        le = LabelEncoder()
        train_enc[col] = train_enc[col].fillna("missing")
        val_enc[col] = val_enc[col].fillna("missing")
        le.fit(train_enc[col])
        val_enc[col] = val_enc[col].map(
            lambda x, le=le: x if x in le.classes_ else "missing"
        )
        le_classes = list(le.classes_)
        if "missing" not in le_classes:
            le_classes.append("missing")
            le = LabelEncoder()
            le.fit(le_classes)
        train_enc[col] = le.transform(train_enc[col])
        val_enc[col] = le.transform(val_enc[col])
        encoders[col] = le

    return train_enc, val_enc, encoders


def run():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(DATA_DIR / "validation.parquet")

    y_train = train[TARGET]
    y_val = validation[TARGET]
    x_train = train.drop(columns=[TARGET])
    x_val = validation.drop(columns=[TARGET])

    x_train_enc, x_val_enc, _ = encode_categoricals(x_train, x_val)

    x_train_enc = x_train_enc.fillna(-1)
    x_val_enc = x_val_enc.fillna(-1)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }

    total = 3 * 3 * 2 * 2
    print(f"Running grid search with 5-fold cross-validation...")
    print(f"Combinations: {total} x 5 folds = {total * 5} fits")

    gs = GridSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(x_train_enc, y_train)

    print(f"\nBest parameters: {gs.best_params_}")
    print(f"Best CV MAE: {-gs.best_score_:.1f} mins")

    best_model = gs.best_estimator_
    y_pred = best_model.predict(x_val_enc)

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
    print(f"XGBoost improvement: {hospital_mae - mae:.1f} mins better")

    importances = pd.Series(
        best_model.feature_importances_, index=x_train_enc.columns
    ).sort_values(ascending=False)
    print(f"\nTop 5 feature importances:")
    for feat, imp in importances.head(5).items():
        print(f"  {feat}: {imp:.3f}")


if __name__ == "__main__":
    run()
