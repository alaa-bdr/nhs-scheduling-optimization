"""Compare cross validation fold counts to find the optimal number.

Dr Aydin asked which fold count performs best on roughly 15,000 rows.
Too few folds gives a noisy estimate, too many is slow and leaves each
fold small. This script tests 3, 5, 7 and 9 folds on the same model and
data, and reports both the mean score and the variability across folds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from nbt_pipeline.modelling_v2.features_v2 import build_v2_dataset

OUTPUT_DIR = Path("data/modelling_v2/plots")
FOLD_COUNTS = [3, 5, 7, 9]
RANDOM_SEED = 42


def encode(frame, features):
    X = frame[features].copy()
    for col in X.select_dtypes(include=["object", "string", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].fillna("missing").astype(str))
    return X.fillna(-1)


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, features, target = build_v2_dataset()
    X = encode(frame, features)
    y = frame[target]

    print(f"Rows: {len(X)}, features: {X.shape[1]}\n")

    rows = []
    for k in FOLD_COUNTS:
        model = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
        )
        cv = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=["r2", "neg_mean_absolute_error"],
            n_jobs=-1, return_train_score=False,
        )
        r2 = scores["test_r2"]
        mae = -scores["test_neg_mean_absolute_error"]
        rows.append({
            "folds": k,
            "mean_R2": r2.mean(),
            "std_R2": r2.std(),
            "min_R2": r2.min(),
            "max_R2": r2.max(),
            "mean_MAE": mae.mean(),
            "std_MAE": mae.std(),
        })
        per_fold = ", ".join(f"{v:.3f}" for v in r2)
        print(f"{k} folds  mean R2={r2.mean():.4f}  sd={r2.std():.4f}  "
              f"mean MAE={mae.mean():.2f}")
        print(f"          per fold R2: {per_fold}")
        best = int(np.argmax(r2)) + 1
        worst = int(np.argmin(r2)) + 1
        print(f"          best fold: {best}, worst fold: {worst}\n")

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "kfold_comparison.csv", index=False)

    optimal = results.loc[results["mean_R2"].idxmax(), "folds"]
    most_stable = results.loc[results["std_R2"].idxmin(), "folds"]
    print(f"Highest mean R2 at {optimal} folds")
    print(f"Most stable (lowest spread) at {most_stable} folds")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].errorbar(results["folds"], results["mean_R2"], yerr=results["std_R2"],
                     marker="o", capsize=5, color="#1B3A5C", linewidth=2)
    axes[0].set_xlabel("Number of folds")
    axes[0].set_ylabel("Mean R2 across folds")
    axes[0].set_title("R2 by fold count (bars show spread)", fontweight="bold")
    axes[0].set_xticks(FOLD_COUNTS)

    axes[1].errorbar(results["folds"], results["mean_MAE"], yerr=results["std_MAE"],
                     marker="o", capsize=5, color="#3366A6", linewidth=2)
    axes[1].set_xlabel("Number of folds")
    axes[1].set_ylabel("Mean absolute error in minutes")
    axes[1].set_title("MAE by fold count (bars show spread)", fontweight="bold")
    axes[1].set_xticks(FOLD_COUNTS)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Cross Validation Fold Count Comparison", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "kfold_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_DIR / 'kfold_comparison.png'}")


if __name__ == "__main__":
    run()
