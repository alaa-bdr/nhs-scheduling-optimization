"""Quantify how much apparent R2 depends on the prediction horizon.

The same dataset supports very different reported accuracy depending on
which fields are admitted as predictors. This script measures that
explicitly by fitting the same model under four horizons, from strictly
preoperative through to fields only knowable after the operation.

The purpose is to establish what R2 is achievable honestly, and to show
how easily a much higher figure can be produced by admitting fields that
would not exist when a theatre list is built.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from nbt_pipeline.modelling_v2.features_v2 import build_v2_dataset

OUTPUT_DIR = Path("data/modelling_v2/plots")
RANDOM_SEED = 42
DURATION_CAP = 480

BEST_PARAMS = dict(
    n_estimators=863, max_depth=10, learning_rate=0.0144,
    subsample=0.6557, colsample_bytree=0.5539,
    min_child_weight=1, reg_alpha=1.1266, reg_lambda=3.9776,
)

INTRAOPERATIVE = [
    "theatre_occupancy_mins",
    "theatre_to_anaesthetic_start_mins",
    "anaesthetic_to_incision_mins",
    "incision_to_closure_mins",
    "closure_to_operation_end_mins",
    "post_operation_theatre_time_mins",
]
OUTCOME_DERIVED = [
    "duration_error_mins",
    "overrun_minutes",
    "underrun_minutes",
    "calculated_operation_length_mins",
]


def encode(frame, features):
    X = frame[features].copy()
    for col in X.select_dtypes(include=["object", "string", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].fillna("missing").astype(str))
    return X.fillna(-1)


def fit_and_score(frame, features, target, label, rows):
    X = encode(frame, features)
    y = frame[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    model = XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbosity=0, **BEST_PARAMS)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    r2 = r2_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    print(f"{label:56} features={len(features):3d}  R2={r2:.4f}  MAE={mae:6.2f}")
    rows.append({"horizon": label, "n_features": len(features), "R2": r2, "MAE": mae})
    return r2


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, base_features, target = build_v2_dataset(duration_cap=DURATION_CAP)
    rows = []

    print(f"Rows after {DURATION_CAP} minute cap: {len(frame)}\n")
    print("Reported accuracy under four prediction horizons\n")

    without_planned = [f for f in base_features if f != "ExpectedDurationMins"]
    fit_and_score(frame, without_planned, target,
                  "1. preoperative, planned duration excluded", rows)

    fit_and_score(frame, base_features, target,
                  "2. preoperative, planned duration included", rows)

    intra = [c for c in INTRAOPERATIVE if c in frame.columns]
    if intra:
        fit_and_score(frame, base_features + intra, target,
                      "3. plus intraoperative stage timings", rows)

    outcome = [c for c in OUTCOME_DERIVED if c in frame.columns]
    if outcome:
        fit_and_score(frame, base_features + intra + outcome, target,
                      "4. plus outcome derived fields", rows)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "leakage_analysis.csv", index=False)

    honest = results.iloc[1]["R2"]
    print(f"\nHonest preoperative ceiling: R2 = {honest:.3f}")
    if len(results) > 2:
        leaked = results.iloc[-1]["R2"]
        print(f"With post scheduling fields admitted: R2 = {leaked:.3f}")
        print(f"Difference attributable to horizon: {leaked - honest:.3f}")

    colours = ["#B0B0B0", "#1B3A5C", "#C97B4E", "#A83232"][:len(results)]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(results["horizon"], results["R2"], color=colours, edgecolor="white", height=0.6)
    for bar, v in zip(bars, results["R2"]):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=10)
    ax.axvline(honest, linestyle="--", color="#1B3A5C", linewidth=1)
    ax.set_xlabel("Test set R2")
    ax.set_xlim(0, 1.05)
    ax.set_title("Reported Accuracy Depends on the Prediction Horizon", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.99, 0.02,
            "Only horizon 2 is usable for scheduling.\nLater horizons use fields that do not exist at listing.",
            transform=ax.transAxes, fontsize=8.5, ha="right", va="bottom",
            style="italic", color="#555555")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "leakage_analysis.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'leakage_analysis.png'}")


if __name__ == "__main__":
    run()
