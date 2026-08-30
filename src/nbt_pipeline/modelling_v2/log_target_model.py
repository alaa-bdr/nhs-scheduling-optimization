"""XGBoost tuned against a log transformed target, with text features.

Operation durations are heavily right skewed, so R2 on raw minutes is
dominated by a small number of very long cases. Modelling log duration
is the standard treatment. Both scales are reported so nothing is hidden.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.stats import randint, uniform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from nbt_pipeline.modelling_v2.features_v2 import build_v2_dataset

OUTPUT_DIR = Path("data/modelling_v2/plots")
SEED = 42
CAP = 480
TEXT_FEATURES = 1000


def encode(frame, features):
    X = frame[features].copy()
    for col in X.select_dtypes(include=["object", "string", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].fillna("missing").astype(str))
    return X.fillna(-1)


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, features, target = build_v2_dataset(duration_cap=CAP)
    print(f"Rows after {CAP} minute cap: {len(frame)}")

    tabular = encode(frame, features)
    notes = frame["theatre_notes"].fillna("").astype(str)
    vectoriser = TfidfVectorizer(
        max_features=TEXT_FEATURES, ngram_range=(1, 2), min_df=3, stop_words="english"
    )
    text = vectoriser.fit_transform(notes)
    X = hstack([csr_matrix(tabular.values), text]).tocsr()
    print(f"Features: {tabular.shape[1]} tabular + {text.shape[1]} text = {X.shape[1]}\n")

    y = frame[target].values
    y_log = np.log1p(y)

    X_tr, X_te, ylog_tr, ylog_te, raw_tr, raw_te = train_test_split(
        X, y_log, y, test_size=0.2, random_state=SEED
    )

    search = RandomizedSearchCV(
        XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0),
        {
            "n_estimators": randint(400, 1400),
            "max_depth": randint(5, 12),
            "learning_rate": uniform(0.01, 0.09),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.4, 0.5),
            "min_child_weight": randint(1, 10),
            "reg_lambda": uniform(0.5, 5.0),
            "reg_alpha": uniform(0.0, 2.0),
        },
        n_iter=40,
        cv=KFold(5, shuffle=True, random_state=SEED),
        scoring="r2",
        n_jobs=-1,
        verbose=1,
        random_state=SEED,
    )
    print("Tuning against the log target, 40 candidates x 5 folds = 200 fits...")
    search.fit(X_tr, ylog_tr)

    print(f"\nBest cross validation R2 on log scale: {search.best_score_:.4f}")

    pred_log = search.best_estimator_.predict(X_te)
    pred_min = np.expm1(pred_log)

    r2_log = r2_score(ylog_te, pred_log)
    r2_min = r2_score(raw_te, pred_min)
    mae_min = mean_absolute_error(raw_te, pred_min)
    rmse_min = mean_squared_error(raw_te, pred_min) ** 0.5

    print("\nTest set results")
    print(f"  R2 on log scale     : {r2_log:.4f}")
    print(f"  R2 on minutes scale : {r2_min:.4f}")
    print(f"  MAE                 : {mae_min:.2f} mins")
    print(f"  RMSE                : {rmse_min:.2f} mins")
    for band in (10, 20, 30, 60):
        print(f"  within {band} mins        : {np.mean(np.abs(raw_te - pred_min) <= band):.1%}")

    planned = frame["ExpectedDurationMins"].values
    _, planned_te = train_test_split(planned, test_size=0.2, random_state=SEED)
    valid = ~np.isnan(planned_te)
    print("\nHospital planned time on the same test rows")
    print(f"  R2 on minutes scale : {r2_score(raw_te[valid], planned_te[valid]):.4f}")
    print(f"  MAE                 : {mean_absolute_error(raw_te[valid], planned_te[valid]):.2f} mins")

    pd.DataFrame([{
        "model": "XGBoost log target with text",
        "R2_log": r2_log, "R2_minutes": r2_min,
        "MAE": mae_min, "RMSE": rmse_min,
    }]).to_csv(OUTPUT_DIR / "log_target_results.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'log_target_results.csv'}")


if __name__ == "__main__":
    run()
