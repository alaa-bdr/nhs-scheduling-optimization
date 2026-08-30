"""Test how outlier handling affects R2 on the v2 feature set.

A small number of very long operations carry most of the unexplained
variance. This script quantifies how much of the reported R2 depends on
how those cases are treated, so the final choice is evidence based and
can be stated openly rather than applied silently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from nbt_pipeline.modelling_v2.features_v2 import build_v2_dataset

OUTPUT_DIR = Path("data/modelling_v2/plots")
RANDOM_SEED = 42

BEST_PARAMS = dict(
    n_estimators=863, max_depth=10, learning_rate=0.0144,
    subsample=0.6557, colsample_bytree=0.5539,
    min_child_weight=1, reg_alpha=1.1266, reg_lambda=3.9776,
)


def encode(frame, features):
    X = frame[features].copy()
    for col in X.select_dtypes(include=["object", "string", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].fillna("missing").astype(str))
    return X.fillna(-1)


def evaluate(cap, log_target, label, rows):
    frame, features, target = build_v2_dataset(duration_cap=cap)
    X = encode(frame, features)
    y = frame[target]
    y_model = np.log1p(y) if log_target else y

    X_tr, X_te, y_tr, y_te, raw_tr, raw_te = train_test_split(
        X, y_model, y, test_size=0.2, random_state=RANDOM_SEED
    )
    model = XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbosity=0, **BEST_PARAMS)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    if log_target:
        pred = np.expm1(pred)

    r2 = r2_score(raw_te, pred)
    mae = mean_absolute_error(raw_te, pred)
    print(f"{label:52} n={len(X):6d}  R2={r2:.4f}  MAE={mae:.2f}")
    rows.append({"setting": label, "n": len(X), "R2": r2, "MAE": mae})


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    print("Effect of capping extreme durations and of a log target\n")
    evaluate(None, False, "no cap, raw target (reference)", rows)
    for cap in (600, 480, 420, 360, 300):
        evaluate(cap, False, f"cap at {cap} mins, raw target", rows)

    print()
    evaluate(None, True, "no cap, log target", rows)
    for cap in (480, 420, 360):
        evaluate(cap, True, f"cap at {cap} mins, log target", rows)

    results = pd.DataFrame(rows).sort_values("R2", ascending=False)
    results.to_csv(OUTPUT_DIR / "outlier_sensitivity.csv", index=False)
    print(f"\nBest setting: {results.iloc[0]['setting']} at R2={results.iloc[0]['R2']:.4f}")
    print(f"Saved: {OUTPUT_DIR / 'outlier_sensitivity.csv'}")


if __name__ == "__main__":
    run()
