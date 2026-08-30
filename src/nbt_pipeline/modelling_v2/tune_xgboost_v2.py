"""Tuned XGBoost on the v2 feature set, evaluated on a held out test set.

Uses a wider randomised hyperparameter search than the v1 round, with
5 fold cross validation chosen on the evidence from kfold_comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from nbt_pipeline.modelling_v2.features_v2 import build_v2_dataset

OUTPUT_DIR = Path("data/modelling_v2/plots")
RANDOM_SEED = 42
N_FOLDS = 5


def encode(frame, features):
    X = frame[features].copy()
    for col in X.select_dtypes(include=["object", "string", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].fillna("missing").astype(str))
    return X.fillna(-1)


def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"{name}")
    print(f"  MAE  = {mae:.2f} mins")
    print(f"  RMSE = {rmse:.2f} mins")
    print(f"  R2   = {r2:.4f}")
    within = {b: float(np.mean(np.abs(y_true - y_pred) <= b)) for b in (10, 20, 30, 60)}
    for b, v in within.items():
        print(f"  within {b} mins: {v:.1%}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, **{f"within_{b}": v for b, v in within.items()}}


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, features, target = build_v2_dataset()
    X = encode(frame, features)
    y = frame[target]

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"Development rows: {len(X_dev)}, test rows: {len(X_test)}\n")

    param_distributions = {
        "n_estimators": randint(300, 1200),
        "max_depth": randint(4, 11),
        "learning_rate": uniform(0.01, 0.14),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(1, 12),
        "reg_lambda": uniform(0.5, 5.0),
        "reg_alpha": uniform(0.0, 2.0),
    }

    search = RandomizedSearchCV(
        XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbosity=0),
        param_distributions,
        n_iter=60,
        cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        scoring="r2",
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_SEED,
    )
    print("Running randomised search, 60 candidates x 5 folds = 300 fits...")
    search.fit(X_dev, y_dev)

    print(f"\nBest cross validation R2: {search.best_score_:.4f}")
    print("Best parameters:")
    for k, v in sorted(search.best_params_.items()):
        print(f"  {k}: {v}")

    print()
    result = report("Tuned XGBoost v2 (test set)", y_test.values, search.best_estimator_.predict(X_test))

    hospital = X_test["ExpectedDurationMins"]
    mask = hospital > 0
    print()
    baseline = report("Hospital planned time (test set)", y_test.values[mask.values], hospital.values[mask.values])

    print(f"\nImprovement over hospital: {baseline['MAE'] - result['MAE']:.2f} mins MAE, "
          f"{result['R2'] - baseline['R2']:.4f} R2")

    pd.DataFrame([result, baseline]).to_csv(OUTPUT_DIR / "xgboost_v2_results.csv", index=False)

    importances = pd.Series(
        search.best_estimator_.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    importances.to_csv(OUTPUT_DIR / "xgboost_v2_importances.csv")
    print("\nTop 10 features:")
    for feat, imp in importances.head(10).items():
        print(f"  {feat}: {imp:.4f}")


if __name__ == "__main__":
    run()
