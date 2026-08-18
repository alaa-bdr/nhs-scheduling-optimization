"""Baseline benchmarks for operation duration prediction.

Two reference points that every later model must beat:
1. The hospital's own planned duration, used as if it were a prediction.
2. A linear regression on the numeric pre-operative features.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path("data/modelling")
TARGET = "operation_length_mins"
NUMERIC_FEATURES = ["ExpectedDurationMins", "age_at_operation", "ASAScore"]


def score(name, y_true, y_pred):
    result = {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }
    print(
        f"{name:<28} MAE={result['MAE']:6.1f}  "
        f"RMSE={result['RMSE']:6.1f}  R2={result['R2']:6.3f}"
    )
    return result


def run():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    validation = pd.read_parquet(DATA_DIR / "validation.parquet")

    y_train = train[TARGET]
    y_validation = validation[TARGET]

    results = []

    hospital = validation.dropna(subset=["ExpectedDurationMins"])
    results.append(
        score(
            "Hospital planned time",
            hospital[TARGET],
            hospital["ExpectedDurationMins"],
        )
    )

    medians = train[NUMERIC_FEATURES].median()
    x_train = train[NUMERIC_FEATURES].fillna(medians)
    x_validation = validation[NUMERIC_FEATURES].fillna(medians)

    model = LinearRegression().fit(x_train, y_train)
    results.append(
        score("Linear regression", y_validation, model.predict(x_validation))
    )

    return pd.DataFrame(results)


if __name__ == "__main__":
    run()
